import pandas as pd
import re
import emoji
from pathlib import Path

# HDFS 관련 함수 임포트 (infra 폴더가 파이썬 경로에 있어야 함)
try:
    # HDFS 클라이언트와 저장 함수만 가져옴
    from infra.hdfs import get_client, save_csv_hdfs
    HDFS_AVAILABLE = True
except ImportError:
    print("[경고] infra.hdfs 모듈을 찾을 수 없습니다. HDFS 저장이 비활성화됩니다.")
    HDFS_AVAILABLE = False

# --- HDFS 저장 경로 설정 (표준 경로 사용) ---
# 운영 파이프라인과 동일한 표준 경로를 사용합니다.
HDFS_OUTPUT_BASE_DIR = "/project-root/data/_1_preprocessed"
# ---------------------------------------------

# 텍스트 정규화 함수
def normalize_text(text: str) -> str:
    """
    텍스트 정규화 함수: HTML 태그, 이모지, 특수문자, URL 제거 및 반복 문자 축소
    """
    if not isinstance(text, str): # 문자열이 아닌 경우 빈 문자열 반환
        return ""
    text = re.sub(r'<[^>]+>', '', text)  # HTML 태그 제거
    text = emoji.replace_emoji(text, replace='')  # 이모지 제거
    text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)  # 특수문자 제거 (한글, 영문, 숫자, 공백 제외)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # 반복 문자 축소 (예: ㅋㅋㅋ -> ㅋㅋ)
    text = re.sub(r"https?:\/\/\S+", "", text) # http, https URL 제거
    text = re.sub(r"www\.\S+", "", text) # www로 시작하는 URL 제거
    return text.strip() # 양 끝 공백 제거

# 종목별 전처리 및 저장 함수
def preprocess_and_save(stock_name: str, base_dir: Path, hdfs_client=None):
    """
    종목별 데이터를 로컬에서 읽어 전처리하고 HDFS 또는 로컬에 저장
    """
    # 입력 파일 경로 (로컬)
    raw_file = base_dir / f"../train_data/_0_raw/{stock_name}_comments.csv"

    # --- 출력 파일 경로 생성 ---
    # HDFS 경로 (표준 경로 사용)
    output_filename_hdfs = f"{stock_name}_preprocessed.csv" # HDFS에 저장될 파일명
    output_path_hdfs = f"{HDFS_OUTPUT_BASE_DIR}/{output_filename_hdfs}"

    # 로컬 경로 (HDFS 사용 불가 시 대체)
    # 참고: HDFS 저장 실패 시 여기에 저장되지만, _4_1_rel_train.py는 HDFS에서 읽으므로 주의
    output_path_local = base_dir / f"../train_data/_1_preprocessed/{stock_name}_preprocessed.csv"
    # --------------------------

    if not raw_file.exists():
        print(f"[❌] 원본 파일 없음: {raw_file}")
        return

    print(f"[INFO] 전처리 시작: {stock_name} ({raw_file})")
    df = pd.read_csv(raw_file)

    # 필요한 컬럼만 선택 (원본 파일에 따라 달라질 수 있음)
    required_cols = ["Message"] # 최소 필요 컬럼
    available_cols = [col for col in ["Comment ID", "Message", "Updated At", "Stock Name"] if col in df.columns]

    if "Message" not in df.columns:
        print(f"[❌] 필수 컬럼 'Message' 없음: {raw_file}")
        return
    elif set(required_cols) != set(available_cols) and len(available_cols) > 1 :
         print(f"[경고] 일부 표준 컬럼이 누락되었습니다. 사용 가능한 컬럼만 유지합니다: {available_cols} ({raw_file})")

    df = df[available_cols] # 사용 가능한 컬럼만 선택

    # Message 컬럼 전처리
    df["Message"] = df["Message"].astype(str).apply(normalize_text)
    # N/A 값 제거
    df = df.dropna(subset=["Message"])
    # 공백 문자열 제거 (요청사항 1 반영)
    df = df[df["Message"].str.strip() != ""]

    if df.empty:
        print(f"[WARN] 전처리 후 유효한 데이터 없음: {stock_name}")
        return

    # HDFS 저장 시도 (요청사항 2 반영)
    if HDFS_AVAILABLE and hdfs_client:
        try:
            # HDFS 디렉토리가 없는 경우 생성 시도 (선택적)
            # hdfs_client.makedirs(HDFS_OUTPUT_BASE_DIR) # 필요하다면 주석 해제
            save_csv_hdfs(hdfs_client, df, output_path_hdfs)
            print(f"[✔] HDFS 저장 완료: {stock_name} → {output_path_hdfs} ({len(df)} 건)")
        except Exception as e:
            print(f"[❌] HDFS 저장 실패: {stock_name} → {e}")
            print(f"[경고] HDFS 저장 실패. 후속 단계(_4_1_rel_train.py)가 데이터를 찾지 못할 수 있습니다.")
            # HDFS 실패 시 로컬 저장 (선택적, 디버깅용)
            # output_path_local.parent.mkdir(parents=True, exist_ok=True)
            # df.to_csv(output_path_local, index=False, encoding="utf-8-sig")
            # print(f"[INFO] 로컬 저장 완료 (대체): {stock_name} → {output_path_local.name} ({len(df)} 건)")
    # HDFS 사용 불가 또는 클라이언트 없음
    else:
        if not HDFS_AVAILABLE:
            print("[경고] HDFS 사용 불가. 후속 단계(_4_1_rel_train.py)가 데이터를 찾지 못할 수 있습니다.")
        else:
            print("[경고] HDFS 클라이언트 없음. 후속 단계(_4_1_rel_train.py)가 데이터를 찾지 못할 수 있습니다.")
        # 로컬 저장 (선택적, 디버깅용)
        # output_path_local.parent.mkdir(parents=True, exist_ok=True)
        # df.to_csv(output_path_local, index=False, encoding="utf-8-sig")
        # print(f"[INFO] 로컬 저장 완료 (대체): {stock_name} → {output_path_local.name} ({len(df)} 건)")


if __name__ == "__main__":
    STOCK_LIST = ['samsung', 'skhynix', 'apple', 'nvidia']
    # 현재 스크립트 파일의 위치를 기준으로 base_dir 설정
    base_dir = Path(__file__).resolve().parent

    # HDFS 클라이언트 초기화 (HDFS 사용 가능 시)
    hdfs_client = None
    if HDFS_AVAILABLE:
        try:
            hdfs_client = get_client()
            print("[INFO] HDFS 클라이언트 연결 성공")
        except Exception as e:
            print(f"[❌] HDFS 클라이언트 연결 실패: {e}")
            HDFS_AVAILABLE = False # 연결 실패 시 HDFS 사용 불가 처리

    # 각 종목에 대해 전처리 및 저장 실행
    for stock_name in STOCK_LIST:
        preprocess_and_save(stock_name, base_dir, hdfs_client)

    print("\n[INFO] 모든 종목 전처리 완료.")
