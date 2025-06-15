import os
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

# Your plain NewsAPI key here (replace with your actual key)
NEWSAPI_KEY = "9067b626da1d4c79a46ee510c93ffec0"

# Folder to save the encrypted key and AES params (same as polygon setup)
SOFTWARE_DIR = r"C:\FPFX\software"

KEY_BIN = os.path.join(SOFTWARE_DIR, "polygon_key.bin")
IV_BIN = os.path.join(SOFTWARE_DIR, "polygon_iv.bin")
ENC_KEY = os.path.join(SOFTWARE_DIR, "api_key.enc")

# Ensure software folder exists
os.makedirs(SOFTWARE_DIR, exist_ok=True)

# Generate random 16-byte AES key and IV
aes_key = os.urandom(16)
aes_iv = os.urandom(16)

# Pad the API key to block size (128 bits = 16 bytes)
padder = padding.PKCS7(128).padder()
padded_data = padder.update(NEWSAPI_KEY.encode()) + padder.finalize()

# Encrypt the padded API key using AES CBC
cipher = Cipher(algorithms.AES(aes_key), modes.CBC(aes_iv), backend=default_backend())
encryptor = cipher.encryptor()
encrypted_key = encryptor.update(padded_data) + encryptor.finalize()

# Save key, IV, and encrypted key to files (overwrite if exist)
with open(KEY_BIN, "wb") as f:
    f.write(aes_key)

with open(IV_BIN, "wb") as f:
    f.write(aes_iv)

with open(ENC_KEY, "wb") as f:
    f.write(encrypted_key)

print(f"NewsAPI key encrypted and saved to:\n - {KEY_BIN}\n - {IV_BIN}\n - {ENC_KEY}")
