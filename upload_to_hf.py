import os
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN", "your_token_here") # Ganti dengan token-mu, jangan dihardcode kalau mau dipush ke github!
api = HfApi()

try:
    user_info = api.whoami(token=token)
    username = user_info['name']
    print(f"✅ Login sukses! Username: {username}")
    
    # -----------------------------------------------------------------
    # UPLOAD KLASIFIKASI
    # -----------------------------------------------------------------
    repo_klasifikasi = f"{username}/Sovereign-Klasifikasi"
    print(f"\n⏳ Membuat/Mengecek repository {repo_klasifikasi}...")
    api.create_repo(repo_id=repo_klasifikasi, token=token, exist_ok=True, private=False)
    
    from pathlib import Path
    folder_klasif = "pesan_klasifikasi/hasil"
    if os.path.exists(folder_klasif):
        print(f"🚀 Mulai mengunggah isi dari '{folder_klasif}' (file per file agar tidak putus)...")
        
        for p in Path(folder_klasif).rglob("*"):
            if p.is_file():
                # path relatif dalam repo
                rel_path = p.relative_to(folder_klasif).as_posix()
                print(f"  -> Mengunggah {rel_path}...")
                try:
                    api.upload_file(
                        path_or_fileobj=str(p),
                        path_in_repo=rel_path,
                        repo_id=repo_klasifikasi,
                        token=token
                    )
                except Exception as ex:
                    print(f"  [!] Gagal mengunggah {rel_path}: {ex}")
                    
        print("🎉 Upload model Klasifikasi SELESAI!")
    else:
        print(f"❌ Folder {folder_klasif} tidak ditemukan.")

except Exception as e:
    print(f"❌ Terjadi kesalahan: {e}")
