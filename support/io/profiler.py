import cProfile
import pstats

def make_profile():
    return cProfile.Profile()

def print_stats(prof):
    prof.disable()
    prof.dump_stats("run_folder_reader.prof")

    stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")

    print("\n=== Top 40 functions overall (cumtime) ===")
    stats.print_stats(40)

    print("\n=== superCalibrateCamera functions ===")
    stats.print_stats("superCalibrateCamera")

    print("\n=== run_folder_reader / analyze_image ===")
    stats.print_stats("run_folder_reader")
    stats.print_stats("analyze_image")

    print("\n=== Top 40 functions overall (cumtime) ===")
    stats.print_stats(40)

    # Narrow view: only functions from your GUI modules
    print("\n=== GUI-ish functions (superCalibrateCamera) ===")
    stats.print_stats("superCalibrateCamera")

    print("\n=== CustomTkinter / Tk wrappers ===")
    stats.print_stats("customtkinter")
    stats.print_stats("ctk")
    stats.print_stats("tkinter")