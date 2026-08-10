import sys
import traceback

sys.path.insert(0, ".")

try:
    import scripts.run_pseudoabsence_ablation_rgcn_transfer as m
    m.main()
except Exception as e:
    with open("err.txt", "w") as f:
        f.write(traceback.format_exc())
    print("CRASH DETECTED:")
    print(traceback.format_exc())
