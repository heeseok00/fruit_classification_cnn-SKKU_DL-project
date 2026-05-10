import torch

print("=" * 50)
print("GPU / CUDA 환경 확인")
print("=" * 50)
print(f"PyTorch 버전     : {torch.__version__}")
print(f"CUDA 사용 가능   : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA 버전        : {torch.version.cuda}")
    print(f"GPU 이름         : {torch.cuda.get_device_name(0)}")
    print(f"GPU 메모리       : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"GPU 개수         : {torch.cuda.device_count()}")

    # 간단한 연산 테스트
    print("\n[GPU 연산 테스트]")
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"행렬 곱셈 결과 shape : {z.shape}")
    print(f"연산 device          : {z.device}")
    print("\n[OK] GPU working correctly!")
else:
    print("\n[FAIL] CUDA not available. Running on CPU.")
