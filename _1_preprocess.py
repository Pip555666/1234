import pandas as pd
import re
import emoji
from pathlib import Path
import sys 

try:
    from infra.hdfs import get_client, save_csv_hdfs
    HDFS_AVAILABLE = True
except ImportError:
    print("[오류] infra.hdfs 모듈을 찾을 수 없습니다. 스크립트를 종료합니다.")
    sys.exit(1) 

HDFS_OUTPUT_BASE_DIR = "/project-root/data/train_preprocessed"

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r"https?:\/\/\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)
    return text.strip()

def preprocess_and_save(stock_name: str, base_dir: Path, hdfs_client):
    raw_file = base_dir / f"../train_data/_0_raw/{stock_name}_comments.csv"

    output_filename_hdfs = f"{stock_name}_preprocessed.csv"
    output_path_hdfs = f"{HDFS_OUTPUT_BASE_DIR}/{output_filename_hdfs}"

    if not raw_file.exists():
        print(f"[❌] 원본 파일 없음: {raw_file}")
        return

    print(f"[INFO] 전처리 시작: {stock_name} ({raw_file})")
    df = pd.read_csv(raw_file)

    required_cols = ["Message"]
    available_cols = [col for col in ["Comment ID", "Message", "Updated At", "Stock Name"] if col in df.columns]

    if "Message" not in df.columns:
        print(f"[❌] 필수 컬럼 'Message' 없음: {raw_file}")
        return
    elif set(required_cols) != set(available_cols) and len(available_cols) > 1 :
         print(f"[경고] 일부 표준 컬럼이 누락되었습니다. 사용 가능한 컬럼만 유지합니다: {available_cols} ({raw_file})")

    df = df[available_cols]

    df["Message"] = df["Message"].astype(str).apply(normalize_text)
    df = df.dropna(subset=["Message"])
    df = df[df["Message"].str.strip() != ""]

    if df.empty:
        print(f"[WARN] 전처리 후 유효한 데이터 없음: {stock_name}")
        return

    try:
        if not hdfs_client.status(HDFS_OUTPUT_BASE_DIR, strict=False):
             hdfs_client.makedirs(HDFS_OUTPUT_BASE_DIR)
             print(f"[INFO] HDFS 디렉토리 생성: {HDFS_OUTPUT_BASE_DIR}")

        save_csv_hdfs(hdfs_client, df, output_path_hdfs)
        print(f"[✔] HDFS 저장 완료: {stock_name} → {output_path_hdfs} ({len(df)} 건)")
    except Exception as e:
        print(f"[❌] HDFS 저장 실패: {stock_name} → {e}")
        raise RuntimeError(f"HDFS 저장 실패: {output_path_hdfs}") from e


if __name__ == "__main__":
    STOCK_LIST = ['samsung', 'skhynix', 'apple', 'nvidia']
    base_dir = Path(__file__).resolve().parent

    hdfs_client = None
    try:
        hdfs_client = get_client()
        print("[INFO] HDFS 클라이언트 연결 성공")
    except Exception as e:
        print(f"[❌] HDFS 클라이언트 연결 실패: {e}")
        sys.exit(1) 
        
    all_successful = True
    for stock_name in STOCK_LIST:
        try:
            preprocess_and_save(stock_name, base_dir, hdfs_client)
        except Exception as proc_e:
            print(f"[오류] {stock_name} 처리 중 오류 발생: {proc_e}")
            all_successful = False

    if all_successful:
        print("\n[INFO] 모든 종목 전처리 및 HDFS 저장 완료.")
    else:
        print("\n[오류] 일부 종목 처리 중 오류가 발생했습니다.")
        sys.exit(1)

