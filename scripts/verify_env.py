import sys

def check_environment():
    packages = [
        ("Python", lambda: sys.version.split()[0]),
        ("PyTorch (torch)", lambda: __import__('torch').__version__),
        ("torchvision", lambda: __import__('torchvision').__version__),
        ("OpenCV (cv2)", lambda: __import__('cv2').__version__),
        ("Pillow (PIL)", lambda: __import__('PIL').__version__),
        ("NumPy (numpy)", lambda: __import__('numpy').__version__),
        ("pandas", lambda: __import__('pandas').__version__),
        ("matplotlib", lambda: __import__('matplotlib').__version__),
        ("scikit-learn (sklearn)", lambda: __import__('sklearn').__version__),
    ]

    print("=" * 60)
    print(f"{'Package / Component':<25} | {'Status':<10} | {'Version'}")
    print("-" * 60)

    all_ok = True
    for name, get_version in packages:
        try:
            ver = get_version()
            print(f"{name:<25} | {'OK':<10} | {ver}")
        except Exception as e:
            print(f"{name:<25} | {'FAILED':<10} | {e}")
            all_ok = False

    print("=" * 60)

    # CUDA / GPU Check for PyTorch
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        print(f"CUDA Available for PyTorch: {cuda_avail}")
        if cuda_avail:
            print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    except Exception:
        pass
    print("=" * 60)

    return all_ok

if __name__ == "__main__":
    check_environment()
