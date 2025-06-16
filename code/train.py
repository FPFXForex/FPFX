# train.py

import os
import math
import random  # Added this import
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import pairwise_distances
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Flatten
import tensorflow as tf
import gym
from gym import spaces
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from tensorflow.keras.optimizers.legacy import Adam
import time
import csv
from collections import defaultdict

# ========== SETUP QUIET ENVIRONMENT ==========
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

# ========== 1) GLOBAL CONFIGURATION ==========
SYMBOLS = ["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","XAUUSD"]
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
DATA_DIR = os.path.join(BASE_DIR,"Data","Processed")
NEWS_CSV = os.path.join(BASE_DIR,"Data","news_cache.csv")
MODEL_DIR = os.path.join(BASE_DIR,"model")
os.makedirs(MODEL_DIR, exist_ok=True)

# Hyperparameters
INITIAL_BALANCE = 100000.0
DISCOUNT_FACTOR = 0.99
TRAIN_STEPS = 200000
MEMORY_LIMIT = 1000000
BATCH_SIZE = 64
OVERTRADE_BARS = 12
MAX_OPEN_TRADES = 3
DAILY_DD_LIMIT = 0.08
MIN_RISK = 0.01
MAX_RISK = 0.02
LOG_INTERVAL = 30           # seconds
MAX_LOT_SIZE = 100.0        # hard cap on lot size

# ========== 2) DATA LOADING ==========
def load_all_data():
    print("\n[1/4] Loading market data...")
    frames=[]
    for sym in SYMBOLS:
        df=pd.read_csv(os.path.join(DATA_DIR,f"{sym}_processed.csv"), parse_dates=["time"])
        df["ATR_14"] *= 10
        df["symbol"]=sym
        frames.append(df)
    all_df=pd.concat(frames,ignore_index=True)
    all_df.sort_values(["time","symbol"],inplace=True)
    all_df.reset_index(drop=True,inplace=True)
    print("[2/4] Merging news sentiment...")
    news=pd.read_csv(NEWS_CSV,parse_dates=["date"])
    if news['date'].dt.tz is None:
        news['date']=news['date'].dt.tz_localize('UTC')
    else:
        news['date']=news['date'].dt.tz_convert('UTC')
    all_df["date"]=all_df["time"].dt.tz_convert('UTC').dt.floor("D")
    all_df=all_df.merge(news,how="left",on=["date","symbol"])
    all_df["news_count"].fillna(0,inplace=True)
    all_df["avg_sentiment"].fillna(0.0,inplace=True)
    all_df.drop(columns=["date"],inplace=True)
    all_df.fillna(0,inplace=True)
    return all_df

ALL_DATA=load_all_data()

print("[3/4] Preparing correlation data...")
H1_CLOSES={}
for sym in SYMBOLS:
    df_sym=ALL_DATA[ALL_DATA["symbol"]==sym].copy()
    df_sym["minute"]=df_sym["time"].dt.minute
    h1_df=df_sym[df_sym["minute"]==0][["time","close"]].copy()
    h1_df.set_index("time",inplace=True)
    H1_CLOSES[sym]=h1_df["close"]

# ========== 3) ENVIRONMENT CLASS ==========
class ForexMultiEnv(gym.Env):
    metadata={"render.modes":["human"]}

    def __init__(self, all_data, initial_balance=INITIAL_BALANCE):
        super().__init__()
        self.all_data=all_data.copy()
        self.symbols=SYMBOLS
        self.n_rows=len(self.all_data)
        self.symbol_to_id={s:i for i,s in enumerate(SYMBOLS)}
        feature_cols=[c for c in self.all_data.columns if c not in ["time","symbol"]]
        self.feature_count=len(feature_cols)
        self.observation_space=spaces.Box(-np.inf,np.inf,
            shape=(self.feature_count+len(SYMBOLS),),dtype=np.float32)
        low=np.array([0.0,0.5,0.5,-0.1,0.0],dtype=np.float32)
        high=np.array([1.0,5.0,5.0,0.1,1.0],dtype=np.float32)
        self.action_space=spaces.Box(low,high,dtype=np.float32)

        self.initial_balance=initial_balance
        self.balance=initial_balance
        self.max_drawdown=0.0
        self.open_positions={}
        self.last_entry_step={s:-OVERTRADE_BARS-1 for s in self.symbols}
        self.features=self.all_data[feature_cols].values
        self.scaler=StandardScaler()
        self.scaler.fit(self.features)
        self.current_step=0
        self.observation=self._get_observation(0)
        self.episode_trades=0
        self.episode_reward=0
        self.total_trades=0
        self.win_count=0
        self.loss_count=0

        # interval summaries
        self.interval_entries=[]
        self.interval_exits=[]
        self.last_progress_report=time.time()

        # prepare trade log CSV
        self.trade_log_path=os.path.join(MODEL_DIR,"trade_log.csv")
        with open(self.trade_log_path,"w",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(["symbol","entry_step","exit_step","duration","lot_size","pnl","confidence"])

    def reset(self):
        self.balance=self.initial_balance
        self.max_drawdown=0.0
        self.open_positions.clear()
        self.last_entry_step={s:-OVERTRADE_BARS-1 for s in self.symbols}
        self.current_step=0
        self.observation=self._get_observation(0)
        self.episode_trades=0
        self.episode_reward=0
        self.total_trades=0
        self.win_count=0
        self.loss_count=0
        self.interval_entries=[]
        self.interval_exits=[]
        self.last_progress_report=time.time()
        with open(self.trade_log_path,"w",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(["symbol","entry_step","exit_step","duration","lot_size","pnl","confidence"])
        return self.observation

    def step(self, action):
        action=np.clip(action,self.action_space.low,self.action_space.high)
        row=self.all_data.iloc[self.current_step]
        symbol=row["symbol"]
        obs=self._get_observation(self.current_step)
        signal, sl_mult, tp_mult, sent_exit_thresh, confidence = action.tolist()
        reward=0.0
        done=False
        info={}

        # dynamic threshold & drawdown check
        current_min_confidence=max(0.25,min(0.65,0.25+(self.episode_trades/1000)))
        dd=(self.initial_balance - self.balance)/self.initial_balance
        if dd>=DAILY_DD_LIMIT:
            return obs, reward, True, info

        # —— TRADE ENTRY —— #
        if confidence>=current_min_confidence and len(self.open_positions)<MAX_OPEN_TRADES and symbol not in self.open_positions:
            t_cur=row["time"]
            h1_time=t_cur.to_pydatetime().replace(minute=0,second=0,microsecond=0)
            def get_last_h1(sym):
                s=H1_CLOSES[sym]
                p=s[s.index<=h1_time]
                return p.values[-50:] if len(p)>=50 else p.values

            skip_corr=False
            for open_sym in self.open_positions:
                a1,a2=get_last_h1(symbol),get_last_h1(open_sym)
                if len(a1)>=2 and len(a2)>=2:
                    m=min(len(a1),len(a2))
                    if np.std(a1[-m:])>0 and np.std(a2[-m:])>0:
                        r=np.corrcoef(a1[-m:],a2[-m:])[0,1]
                        if abs(r)>0.8:
                            skip_corr=True
                            break

            if not skip_corr:
                risk_pct=MIN_RISK+confidence*(MAX_RISK-MIN_RISK)
                risk_amt=risk_pct*self.initial_balance
                atr=row["ATR_14"] if row["ATR_14"]>0 else 1e-6
                pip_value=0.01 if (symbol.endswith("JPY") or symbol=="XAUUSD") else 0.0001
                sl_pips=atr*sl_mult if atr*sl_mult>0 else atr
                raw_lot=risk_amt/(sl_pips/pip_value)
                lot_size=float(min(max(raw_lot,0.01),MAX_LOT_SIZE))
                print(f"[DEBUG LOT] {symbol} | risk_amt={risk_amt:.2f} | sl_pips={sl_pips:.6f} | pip_val={pip_value} | raw_lot={raw_lot:.2f} | capped={lot_size:.2f}")

                direction=1 if signal>=0.5 else -1
                entry_price=row["open"]
                sl_price=entry_price-direction*sl_pips
                tp_price=entry_price+direction*(atr*tp_mult)

                self.open_positions[symbol]={
                    "direction":direction,
                    "entry_price":entry_price,
                    "sl_price":sl_price,
                    "tp_price":tp_price,
                    "lot_size":lot_size,
                    "entry_step":self.current_step,
                    "confidence":confidence,
                    "sent_exit_thresh":sent_exit_thresh
                }
                self.episode_trades+=1
                self.total_trades+=1
                self.interval_entries.append({
                    "symbol":symbol,
                    "entry_step":self.current_step,
                    "lot_size":lot_size,
                    "confidence":confidence
                })
                reward+=0.002

        # —— TRADE EXIT —— #
        if symbol in self.open_positions:
            pos=self.open_positions[symbol]
            direction,entry_price,sl_price,tp_price,lot_size,sent_thresh = \
                pos["direction"],pos["entry_price"],pos["sl_price"],pos["tp_price"],pos["lot_size"],pos["sent_exit_thresh"]
            pip_value=0.01 if (symbol.endswith("JPY") or symbol=="XAUUSD") else 0.0001
            low,high,close=row["low"],row["high"],row["close"]
            pnl=0.0
            closed=False

            if (direction==1 and low<=sl_price) or (direction==-1 and high>=sl_price):
                exit_price=sl_price; pnl=(exit_price-entry_price)*direction*(lot_size/pip_value); closed=True
            elif (direction==1 and high>=tp_price) or (direction==-1 and low<=tp_price):
                exit_price=tp_price; pnl=(exit_price-entry_price)*direction*(lot_size/pip_value); closed=True
            elif (direction==1 and row["avg_sentiment"]<=sent_thresh) or \
                 (direction==-1 and row["avg_sentiment"]>=sent_thresh):
                exit_price=close; pnl=(exit_price-entry_price)*direction*(lot_size/pip_value); closed=True

            if closed:
                overtrade_pen=abs(pnl)*0.25 if (self.current_step-pos["entry_step"]<OVERTRADE_BARS) else 0.0
                self.balance+=pnl
                ddc=(self.initial_balance-self.balance)/self.initial_balance
                if ddc>self.max_drawdown: self.max_drawdown=ddc
                draw_pen=1.7*(-pnl if pnl<0 else 0.0)
                reward+=pnl-overtrade_pen-draw_pen

                duration=self.current_step-pos["entry_step"]
                self.interval_exits.append({
                    "symbol":symbol,
                    "entry_step":pos["entry_step"],
                    "exit_step":self.current_step,
                    "duration":duration,
                    "lot_size":lot_size,
                    "pnl":pnl,
                    "confidence":pos["confidence"]
                })
                # CSV log
                with open(self.trade_log_path,"a",newline="") as f:
                    w=csv.writer(f)
                    w.writerow([symbol,pos["entry_step"],self.current_step,duration,
                                f"{lot_size:.2f}",f"{pnl:.2f}",f"{pos['confidence']:.3f}"])
                del self.open_positions[symbol]
                if pnl>0: self.win_count+=1
                else: self.loss_count+=1

        # small penalty
        if not self.open_positions and random.random()<0.1:
            reward-=0.001

        # advance
        self.current_step+=1
        if self.current_step>=self.n_rows:
            done=True
            next_obs=np.zeros(self.observation_space.shape,dtype=np.float32)
        else:
            done=False
            next_obs=self._get_observation(self.current_step)

        info.update({
            "balance":self.balance,
            "max_drawdown":self.max_drawdown,
            "open_positions":len(self.open_positions),
            "episode_trades":self.episode_trades
        })

        # periodic summary
        now=time.time()
        if now-self.last_progress_report>=LOG_INTERVAL:
            pct=(self.current_step/self.n_rows)*100
            print(f"\n[Summary] Step {self.current_step}/{self.n_rows} ({pct:.1f}%) "
                  f"| Bal: ${self.balance:,.2f} | DD: {self.max_drawdown*100:.1f}% "
                  f"| Trades: {self.total_trades} | Wins: {self.win_count} | Losses: {self.loss_count}")
            # entries summary
            if self.interval_entries:
                print(" Entries last interval:")
                by_sym=defaultdict(list)
                for e in self.interval_entries: by_sym[e["symbol"]].append(e)
                for s, lst in by_sym.items():
                    avg_conf=np.mean([e["confidence"] for e in lst])
                    avg_lot=np.mean([e["lot_size"] for e in lst])
                    print(f"  • {s}: {len(lst)} entries, avg lot={avg_lot:.2f}, avg conf={avg_conf:.3f}")
            # exits summary
            if self.interval_exits:
                print(" Exits last interval:")
                by_sym=defaultdict(list)
                for e in self.interval_exits: by_sym[e["symbol"]].append(e)
                for s,lst in by_sym.items():
                    avg_pnl=np.mean([e["pnl"] for e in lst])
                    avg_dur=np.mean([e["duration"] for e in lst])
                    print(f"  • {s}: {len(lst)} exits, avg pnl={avg_pnl:.2f}, avg dur={avg_dur:.1f} steps")
            self.interval_entries=[]
            self.interval_exits=[]
            self.last_progress_report=now

        self.episode_reward+=reward
        return next_obs, reward, done, info

    def _get_observation(self, step):
        row=self.all_data.iloc[step]
        feat=row.drop(["time","symbol"]).values.astype(np.float32)
        scaled=self.scaler.transform(feat.reshape(1,-1)).flatten().astype(np.float32)
        onehot=np.zeros(len(SYMBOLS),dtype=np.float32)
        onehot[self.symbol_to_id[row["symbol"]]]=1.0
        return np.concatenate([scaled,onehot])

    def render(self, mode="human"): pass
    def close(self): pass

# ========== 4) MODEL ARCHITECTURES ==========
def build_actor(input_shape, action_space):
    state_input=Input(shape=input_shape,name="state_input")
    x=Flatten()(state_input)
    x=Dense(256,activation="relu")(x)
    x=Dense(256,activation="relu")(x)
    x=Dense(128,activation="relu")(x)
    raw=Dense(action_space.shape[0],activation="sigmoid",name="raw_actions")(x)
    lb,hb=action_space.low,action_space.high
    actions=tf.keras.layers.Lambda(lambda a: a*(hb-lb)+lb,name="actions")(raw)
    return Model(inputs=state_input,outputs=actions)

def build_critic(input_shape, action_space):
    state_input=Input(shape=input_shape,name="state_input")
    x=Flatten()(state_input)
    action_input=Input(shape=(action_space.shape[0],),name="action_input")
    xs=Dense(256,activation="relu")(x)
    xa=Concatenate()([xs,action_input])
    x=Dense(256,activation="relu")(xa)
    x=Dense(128,activation="relu")(x)
    q= Dense(1,activation="linear",name="q_value")(x)
    return Model(inputs=[state_input,action_input],outputs=q)

# ========== 5) TRAINING ROUTINE ==========
def train_agent():
    print("\n[4/4] Initializing training environment...")
    env=ForexMultiEnv(ALL_DATA,initial_balance=INITIAL_BALANCE)
    nb_actions=env.action_space.shape[0]
    scaler_path=os.path.join(MODEL_DIR,"scaler.pkl")
    joblib.dump(env.scaler,scaler_path)
    print(f"✓ Scaler saved to {scaler_path}")

    actor=build_actor((1,)+env.observation_space.shape,env.action_space)
    critic=build_critic((1,)+env.observation_space.shape,env.action_space)
    memory=SequentialMemory(limit=MEMORY_LIMIT,window_length=1)
    rp=OrnsteinUhlenbeckProcess(size=nb_actions,theta=0.25,mu=0.0,sigma=0.3,sigma_min=0.05)
    agent=DDPGAgent(nb_actions=nb_actions,actor=actor,critic=critic,
                    critic_action_input=critic.input[1],memory=memory,
                    nb_steps_warmup_critic=5000,nb_steps_warmup_actor=5000,
                    random_process=rp,gamma=DISCOUNT_FACTOR,
                    target_model_update=1e-3,batch_size=BATCH_SIZE)
    agent.compile(Adam(1e-4,clipnorm=1.0),metrics=["mae"])

    print(f"\n▶ Starting training for {TRAIN_STEPS} steps...")
    print(" Progress summary every 30s")
    agent.fit(env,nb_steps=TRAIN_STEPS,visualize=False,verbose=2,nb_max_episode_steps=env.n_rows)

    actor_path=os.path.join(MODEL_DIR,"actor_model.h5")
    critic_path=os.path.join(MODEL_DIR,"critic_model.h5")
    weights_path=os.path.join(MODEL_DIR,"ddpg_weights.h5f")
    actor.save(actor_path)
    critic.save_weights(critic_path)
    agent.save_weights(weights_path,overwrite=True)

    print("\n=== Training Summary ===")
    print(f"Final balance: ${env.balance:,.2f}")
    print(f"Total trades: {env.total_trades} | Wins: {env.win_count} | Losses: {env.loss_count}")
    print(f"Max drawdown: {env.max_drawdown*100:.2f}%")
    print(f"✓ Trade log: {env.trade_log_path}")
    print("✓ Models saved.")
    
if __name__=="__main__":
    gpus=tf.config.list_physical_devices('GPU')
    if gpus: tf.config.experimental.set_memory_growth(gpus[0],True)
    try:
        train_agent()
    except Exception as e:
        print(f"\n⚠️ Training failed: {e}")
        print("Check data files and GPU availability")
