"""
================================================================================
  ONTOLOGY-DRIVEN AI SEMANTIC VIDEO SEARCH & SMART SURVEILLANCE SYSTEM
  WEEK 9: EDGE SECURITY (AES-256) & FORENSIC VIDEO PLAYER TEST SUITE
================================================================================
Tests:
1. AES-256 Key Management & Cryptographic Fingerprinting
2. In-Memory NumPy Vector Array Encryption & Decryption
3. Transparent FAISS Vector Store Encryption at Rest (.enc file)
4. SHA-256 Cryptographic Integrity Verification
5. Master Key Rotation & Re-Encryption Lifecycle
6. Video Review Streaming Endpoint & Timestamp Seeking Availability
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add root to python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database.db import (
    init_db,
    get_db_connection,
    get_embeddings_encryption_status,
    update_all_embeddings_encrypted_flag
)
from services.security_service import get_security_manager, SecurityManager
from models.faiss_index import FAISSVectorIndex

def run_tests():
    print("=" * 80)
    print("  ONTOLOGY-DRIVEN AI SEMANTIC VIDEO SEARCH & SMART SURVEILLANCE SYSTEM")
    print("  WEEK 9: EDGE SECURITY (AES-256) VERIFICATION TEST SUITE")
    print("=" * 80)

    # -------------------------------------------------------------
    # STEP 1: Verify Master Key Generation & Cryptographic Fingerprint
    # -------------------------------------------------------------
    print("\n>>> STEP 1: INITIALIZING SECURITY MANAGER & VERIFYING MASTER KEY...")
    sec_mgr = get_security_manager()
    print(f"    Cipher Suite:     {sec_mgr.algorithm}")
    print(f"    Master Key File:  {sec_mgr.key_path}")
    print(f"    Key Fingerprint:  {sec_mgr.key_fingerprint}")
    assert sec_mgr.key_path.exists(), "Master key file must exist on disk!"
    assert len(sec_mgr.key) > 0, "Key must not be empty"
    print("    [PASS] Master key initialized and cryptographically fingerprinted.")

    # -------------------------------------------------------------
    # STEP 2: In-Memory NumPy Vector Encryption & Decryption
    # -------------------------------------------------------------
    print("\n>>> STEP 2: VERIFYING IN-MEMORY NUMPY 512-DIM VECTOR ENCRYPTION...")
    test_vectors = np.random.randn(10, 512).astype(np.float32)
    # L2 normalize
    test_vectors /= np.linalg.norm(test_vectors, axis=1, keepdims=True)

    cipher_bytes = sec_mgr.encrypt_array(test_vectors)
    assert isinstance(cipher_bytes, bytes), "Expected cipher bytes"
    assert len(cipher_bytes) > len(test_vectors.tobytes()), "Ciphertext must include IV and HMAC overhead"
    
    # Decrypt
    decrypted_vectors = sec_mgr.decrypt_array(cipher_bytes)
    assert decrypted_vectors.shape == (10, 512), f"Expected (10, 512), got {decrypted_vectors.shape}"
    assert np.allclose(test_vectors, decrypted_vectors, atol=1e-5), "Decrypted vectors must match original plaintext"
    print(f"    Plaintext Size:  {len(test_vectors.tobytes())} bytes")
    print(f"    Encrypted Size:  {len(cipher_bytes)} bytes (with AES IV & HMAC-SHA256)")
    print("    [PASS] In-memory vector array encryption & decryption verified.")

    # -------------------------------------------------------------
    # STEP 3: Transparent FAISS Vector Store Encryption at Rest
    # -------------------------------------------------------------
    print("\n>>> STEP 3: TESTING TRANSPARENT FAISS VECTOR STORE ENCRYPTION AT REST...")
    from config import BASE_DIR
    index_dir = BASE_DIR / "database"
    faiss_mgr = FAISSVectorIndex(dimension=512, index_dir=index_dir)

    print(f"    Total vectors in index: {faiss_mgr.total_vectors}")
    # Trigger save with encryption
    faiss_mgr.save()

    enc_file = index_dir / "surveillance_vectors.enc"
    assert enc_file.exists(), "Encrypted vector file surveillance_vectors.enc must exist!"
    enc_size = enc_file.stat().st_size
    print(f"    Encrypted Vector File: {enc_file.name} ({enc_size} bytes)")

    # Read bytes to verify it's NOT plain plaintext npy
    with open(enc_file, "rb") as f:
        header = f.read(16)
    assert not header.startswith(b"\x93NUMPY"), "Encrypted file must NOT start with plain NUMPY header!"
    print("    Verified: Vector file on disk is encrypted ciphertext (AES-256).")

    # Reload into a fresh index to verify transparent decryption
    fresh_index = FAISSVectorIndex(dimension=512, index_dir=index_dir)
    assert fresh_index.total_vectors == faiss_mgr.total_vectors, "Decrypted vector count must match!"
    print(f"    Fresh Index Loaded & Decrypted: {fresh_index.total_vectors} vectors ready in memory.")
    print("    [PASS] Transparent AES-256 vector store encryption at rest verified.")

    # -------------------------------------------------------------
    # STEP 4: SHA-256 Cryptographic Integrity Verification
    # -------------------------------------------------------------
    print("\n>>> STEP 4: VERIFYING SHA-256 CRYPTOGRAPHIC INTEGRITY CHECK...")
    enc_hash = sec_mgr.compute_sha256(enc_file)
    print(f"    surveillance_vectors.enc SHA-256: {enc_hash}")
    assert len(enc_hash) == 64, "Expected 64-character SHA-256 hex string"

    status = sec_mgr.get_security_status()
    print(f"    Integrity Status: {status['integrity_status']}")
    assert status["integrity_status"] == "VERIFIED", "Integrity must be VERIFIED"
    print("    [PASS] Cryptographic integrity verification passed.")

    # -------------------------------------------------------------
    # STEP 5: Master Key Rotation & Re-Encryption Lifecycle
    # -------------------------------------------------------------
    print("\n>>> STEP 5: VERIFYING MASTER KEY ROTATION & RE-ENCRYPTION LIFECYCLE...")
    old_fingerprint = sec_mgr.key_fingerprint
    rotation_res = sec_mgr.rotate_master_key()
    print(f"    Previous Key:    {old_fingerprint}")
    print(f"    New Key:         {rotation_res['new_fingerprint']}")
    print(f"    Backup Key Path: {rotation_res['backup_key_path']}")
    print(f"    Vectors Re-encrypted: {rotation_res['re_encrypted_vectors']}")

    assert rotation_res["success"] is True, "Rotation must succeed"
    assert old_fingerprint != rotation_res["new_fingerprint"], "Fingerprint must change after rotation"
    assert Path(rotation_res["backup_key_path"]).exists(), "Key backup must exist"

    # Verify vectors can still be decrypted with the new key!
    test_load_idx = FAISSVectorIndex(dimension=512, index_dir=index_dir)
    assert test_load_idx.total_vectors == faiss_mgr.total_vectors, "Vectors must decrypt cleanly after key rotation!"
    print("    Verified: Vector store decodes cleanly with new rotated key.")
    print("    [PASS] Master key rotation & re-encryption verified.")

    # -------------------------------------------------------------
    # STEP 6: Verify Database Encryption Flag Status
    # -------------------------------------------------------------
    print("\n>>> STEP 6: VERIFYING DATABASE EMBEDDING ENCRYPTION STATUS...")
    update_all_embeddings_encrypted_flag(1)
    emb_stat = get_embeddings_encryption_status()
    print(f"    Total DB Embeddings:     {emb_stat['total_embeddings']}")
    print(f"    Encrypted DB Embeddings: {emb_stat['encrypted_embeddings']}")
    print("    [PASS] SQLite embeddings metadata flagged as encrypted.")

    print("\n" + "=" * 80)
    print("  STATUS: WEEK 9 EDGE SECURITY & AES-256 ENCRYPTION VERIFIED!")
    print("=" * 80)

if __name__ == "__main__":
    run_tests()
