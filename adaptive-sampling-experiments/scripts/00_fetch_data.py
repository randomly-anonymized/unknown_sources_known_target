"""Download the NHIS 2023 public-use files this project needs into data/raw/.

Sample Adult : https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2023/adult23csv.zip
Paradata     : https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2023/paradata23csv.zip
Documentation: https://ftp.cdc.gov/pub/health_Statistics/NCHs/Dataset_Documentation/NHIS/2023/

Both files are public domain (NCHS). If this machine has no network, download them by hand and
put adult23.csv and paradata23.csv in data/raw/ (or point $NHIS_DIR at them).
"""
import hashlib, io, os, sys, urllib.request, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
FILES = {
    "adult23.csv": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2023/adult23csv.zip",
    "paradata23.csv": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2023/paradata23csv.zip",
}


def main():
    os.makedirs(RAW, exist_ok=True)
    for name, url in FILES.items():
        out = os.path.join(RAW, name)
        if os.path.exists(out):
            print(f"have {name}")
            continue
        print(f"downloading {url}")
        try:
            blob = urllib.request.urlopen(url, timeout=120).read()
        except Exception as e:
            sys.exit(f"download failed ({e}). Fetch {url} manually and unzip into {RAW}.")
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            member = [m for m in z.namelist() if m.lower().endswith(".csv")][0]
            open(out, "wb").write(z.read(member))
        print(f"  wrote {out} ({os.path.getsize(out):,} bytes, "
              f"sha256[:16]={hashlib.sha256(open(out,'rb').read()).hexdigest()[:16]})")


if __name__ == "__main__":
    main()
