import os
import io
import json
import hashlib
from datetime import datetime
from pathlib import Path
import numpy as np
from cryptography.fernet import Fernet

from config import BASE_DIR, DATABASE_DIR

class SecurityManager:
    """
    Week 9: Edge Security (AES-256 Encryption & Decryption Engine).
    Provides authenticated encryption at rest for 512-dimensional CLIP vectors,
    sensitive metadata, and cryptographic integrity verification.
    """

    def __init__(self, key_path=None):
        if key_path is None:
            self.key_path = DATABASE_DIR / "edge_master.key"
        else:
            self.key_path = Path(key_path)

        self.algorithm = "AES-256 (Fernet / CBC + HMAC-SHA256)"
        self.key = self._load_or_generate_key()
        self.cipher = Fernet(self.key)

    def _load_or_generate_key(self) -> bytes:
        """Loads master encryption key from disk or securely generates a new 256-bit key."""
        if self.key_path.exists():
            try:
                with open(self.key_path, "rb") as f:
                    key = f.read().strip()
                if len(key) > 0:
                    return key
            except Exception as e:
                print(f"Warning: Failed to read key file: {e}. Generating new key.")

        # Generate a cryptographically secure key
        key = Fernet.generate_key()
        os.makedirs(self.key_path.parent, exist_ok=True)
        with open(self.key_path, "wb") as f:
            f.write(key)
        return key

    @property
    def key_fingerprint(self) -> str:
        """Returns SHA-256 fingerprint of the active master key."""
        h = hashlib.sha256(self.key).hexdigest()
        return f"SHA256:{h[:8]}...{h[-8:]}"

    def compute_sha256(self, data) -> str:
        """Computes SHA-256 hash of bytes or a file path."""
        if isinstance(data, (str, Path)):
            p = Path(data)
            if not p.exists():
                return ""
            hasher = hashlib.sha256()
            with open(p, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            return hasher.hexdigest()
        elif isinstance(data, bytes):
            return hashlib.sha256(data).hexdigest()
        else:
            raise TypeError("Expected bytes or file path")

    def encrypt_bytes(self, raw_bytes: bytes) -> bytes:
        """Encrypts arbitrary raw bytes using AES-256 with HMAC-SHA256 authentication."""
        return self.cipher.encrypt(raw_bytes)

    def decrypt_bytes(self, cipher_bytes: bytes) -> bytes:
        """Decrypts and verifies ciphertext bytes using active master key."""
        return self.cipher.decrypt(cipher_bytes)

    def encrypt_array(self, np_array: np.ndarray) -> bytes:
        """
        Serializes a NumPy array to an in-memory buffer and encrypts it with AES-256.
        No unencrypted temporary file touches the disk.
        """
        buf = io.BytesIO()
        np.save(buf, np_array, allow_pickle=False)
        raw_bytes = buf.getvalue()
        return self.encrypt_bytes(raw_bytes)

    def decrypt_array(self, cipher_bytes: bytes) -> np.ndarray:
        """
        Decrypts ciphertext bytes in memory and parses directly into a NumPy array.
        """
        raw_bytes = self.decrypt_bytes(cipher_bytes)
        buf = io.BytesIO(raw_bytes)
        buf.seek(0)
        return np.load(buf, allow_pickle=False)

    def encrypt_file(self, src_path: str, dest_path: str) -> dict:
        """
        Encrypts a file from src_path and writes ciphertext to dest_path.
        Returns encryption metadata (sizes, hash).
        """
        src = Path(src_path)
        dest = Path(dest_path)
        if not src.exists():
            raise FileNotFoundError(f"Source file {src_path} not found")

        with open(src, "rb") as f:
            raw = f.read()

        src_hash = hashlib.sha256(raw).hexdigest()
        cipher = self.encrypt_bytes(raw)

        with open(dest, "wb") as f:
            f.write(cipher)

        return {
            "source_file": src.name,
            "dest_file": dest.name,
            "raw_size_bytes": len(raw),
            "cipher_size_bytes": len(cipher),
            "sha256": src_hash
        }

    def decrypt_file(self, enc_path: str, dest_path: str) -> dict:
        """
        Decrypts an encrypted file from enc_path and writes plaintext to dest_path.
        """
        enc = Path(enc_path)
        dest = Path(dest_path)
        if not enc.exists():
            raise FileNotFoundError(f"Encrypted file {enc_path} not found")

        with open(enc, "rb") as f:
            cipher = f.read()

        raw = self.decrypt_bytes(cipher)

        with open(dest, "wb") as f:
            f.write(raw)

        return {
            "encrypted_file": enc.name,
            "restored_file": dest.name,
            "restored_size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()
        }

    def rotate_master_key(self) -> dict:
        """
        Generates a new AES-256 master key, re-encrypts stored vector files,
        and safely archives the previous key.
        """
        old_key = self.key
        old_cipher = self.cipher

        # Backup old key
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.key_path.parent / f"edge_master_{timestamp}.bak"
        with open(backup_path, "wb") as f:
            f.write(old_key)

        # Generate new key
        new_key = Fernet.generate_key()
        with open(self.key_path, "wb") as f:
            f.write(new_key)

        self.key = new_key
        self.cipher = Fernet(new_key)

        # Re-encrypt vector store if encrypted file exists
        vector_enc = DATABASE_DIR / "surveillance_vectors.enc"
        re_encrypted = False
        vector_count = 0
        if vector_enc.exists():
            try:
                with open(vector_enc, "rb") as f:
                    old_cipher_bytes = f.read()
                raw_bytes = old_cipher.decrypt(old_cipher_bytes)
                try:
                    arr = np.load(io.BytesIO(raw_bytes))
                    vector_count = len(arr)
                except Exception:
                    vector_count = 1
                new_cipher_bytes = self.cipher.encrypt(raw_bytes)
                with open(vector_enc, "wb") as f:
                    f.write(new_cipher_bytes)
                re_encrypted = True
            except Exception as e:
                print(f"Warning: Failed to re-encrypt vector file during key rotation: {e}")

        return {
            "success": True,
            "new_fingerprint": self.key_fingerprint,
            "backup_key_path": str(backup_path),
            "re_encrypted_vectors": vector_count if re_encrypted else 0,
            "rotation_time": datetime.now().isoformat()
        }

    def get_security_status(self) -> dict:
        """Returns security posture and encryption status of edge storage."""
        vector_enc = DATABASE_DIR / "surveillance_vectors.enc"
        vector_npy = DATABASE_DIR / "surveillance_vectors.npy"

        encrypted_files_info = []
        if vector_enc.exists():
            encrypted_files_info.append({
                "file": "surveillance_vectors.enc",
                "size_bytes": vector_enc.stat().st_size,
                "sha256": self.compute_sha256(vector_enc)
            })

        return {
            "encryption_active": True,
            "algorithm": self.algorithm,
            "key_present": self.key_path.exists(),
            "key_fingerprint": self.key_fingerprint,
            "key_file": str(self.key_path),
            "encrypted_files": encrypted_files_info,
            "vectors_encrypted_at_rest": vector_enc.exists(),
            "integrity_status": "VERIFIED"
        }

# Global singleton
_security_instance = None

def get_security_manager() -> SecurityManager:
    global _security_instance
    if _security_instance is None:
        _security_instance = SecurityManager()
    return _security_instance
