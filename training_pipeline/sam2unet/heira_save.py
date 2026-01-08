import os
import urllib.request

# 폴더 생성
os.makedirs("checkpoints", exist_ok=True)

# 다운로드
url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
save_path = "checkpoints/sam2_hiera_large.pt"

print("다운로드 시작... (시간이 조금 걸릴 수 있습니다)")
urllib.request.urlretrieve(url, save_path)
print("다운로드 완료!")