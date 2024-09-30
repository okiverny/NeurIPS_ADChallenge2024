"""
Calibrating script for the Airel Data Challenge 2024
"""

import ctypes
import gc
import multiprocessing
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import rich.progress
from astropy.stats import sigma_clip

# Cut off frequencies for AIRS data
CUT_INF = 39
CUT_SUP = 321

class Progress_bar(rich.progress.Progress):
    def __init__(self):
        super().__init__(
            rich.progress.BarColumn(),
            rich.progress.TextColumn("[green]{task.percentage:>3.0f}%"),
            rich.progress.TextColumn("•"),
            rich.progress.TimeElapsedColumn(),
            rich.progress.TextColumn("•"),
            rich.progress.TimeRemainingColumn(),
            "•[magenta] {task.completed}/{task.total} processed",
        )

verbose_mode = {
    0: None,
    1: "progress_bar",
    2: "print",
}

class Calibrator:
    def __init__(
        self,
        is_test: bool,
        data_dir: str,
        output_dir: str,
        c_lib_path: str,
        num_workers: int = 1,
        time_binning_freq: int = 30,
        first_n_files: int = 0,
        verbose: int = 0,
    ):
        if is_test:
            self.test_train_prefix = "test"
        else:
            self.test_train_prefix = "train"
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.c_lib_path = c_lib_path
        self.num_workers = num_workers
        self.time_binning_freq = time_binning_freq
        self.first_n_files = first_n_files
        self.verbose = verbose

        print("Calibration library for Ariel Data Challenge 2024")
        print("=====================================================================")
        print("Calibrate test dataset:", is_test)
        print("Data directory:", self.data_dir)
        print("Output directory:", self.output_dir)
        print("C library path:", self.c_lib_path)
        print("Number of workers:", self.num_workers)
        print("Time binning frequency:", self.time_binning_freq)
        print("First n files:", self.first_n_files)
        print("Verbose:", verbose_mode[self.verbose])
        print("=====================================================================")

    def calibrate(self):
        print("Calibrating the data...")
        files = list((self.data_dir / self.test_train_prefix).glob("*/*"))
        self.indices_ = self.get_index(files)
        if self.first_n_files > 0:
            self.indices_ = self.indices_[: self.first_n_files]

        adc_info = pd.read_csv(self.data_dir / f"{self.test_train_prefix}_adc_info.csv")
        adc_info = adc_info.set_index("planet_id")

        axis_info = pd.read_parquet(self.data_dir / "axis_info.parquet")
        dt_airs = axis_info["AIRS-CH0-integration_time"].dropna().values.copy()
        dt_airs[1::2] += 0.1

        # Benchmarking
        start = timeit.default_timer()
        with multiprocessing.Pool(self.num_workers) as pool:
            tasks = [
                (
                    self.c_lib_path,
                    n,
                    planet_id,
                    self.data_dir,
                    self.output_dir,
                    self.test_train_prefix,
                    self.time_binning_freq,
                    adc_info,
                    dt_airs,
                )
                for n, planet_id in enumerate(self.indices_)
            ]
            
            match self.verbose:
                case 0:
                    pool.imap_unordered(self._calibrate_one_data_wrapper, tasks)
                case 1:
                    progress_bar = Progress_bar()
                    progress_bar_task = progress_bar.add_task("", total=len(tasks))
                    with progress_bar:
                        for _ in pool.imap_unordered(self._calibrate_one_data_wrapper, tasks):
                            progress_bar.update(progress_bar_task, advance=1)
                case 2:
                    processed = 0
                    for _ in pool.imap_unordered(self._calibrate_one_data_wrapper, tasks):
                        processed += 1
                        print(f"Processed {processed}/{len(tasks)}, Elapsed time: {timeit.default_timer() - start:.1f} s")

        end = timeit.default_timer()
        print(f"All done! Elapsed time: {end - start:.1f} s")
        print()
    
    @staticmethod
    def _calibrate_one_data_wrapper(args):
        return Calibrator._calibrate_one_data(*args)

    @staticmethod
    def _calibrate_one_data(
        c_lib_path: str,
        n: int,
        planet_id: int,
        data_dir: Path,
        output_dir: Path,
        test_train_prefix: str,
        time_binning_freq: int,
        adc_info: pd.DataFrame,
        dt_airs: np.ndarray,
    ):
        c_lib = ctypes.cdll.LoadLibrary(c_lib_path)

        ### AIRS ###
        signal = pl.read_parquet(
            data_dir / f"{test_train_prefix}/{planet_id}/AIRS-CH0_signal.parquet"
        )
        signal = (
            signal.to_numpy().astype(np.float64).reshape((signal.shape[0], 32, 356))
        )
        signal = signal.copy()
        gain = adc_info["AIRS-CH0_adc_gain"].loc[planet_id]
        offset = adc_info["AIRS-CH0_adc_offset"].loc[planet_id]

        linear_corr = (
            pl.read_parquet(
                data_dir
                / f"{test_train_prefix}/{planet_id}/AIRS-CH0_calibration/linear_corr.parquet"
            )
            .to_numpy()
            .astype(np.float64)
            .reshape((6, 32, 356))
        ).copy()
        dark = (
            pl.read_parquet(
                data_dir
                / f"{test_train_prefix}/{planet_id}/AIRS-CH0_calibration/dark.parquet"
            )
            .to_numpy()
            .astype(np.float64)
            .reshape((32, 356))
        ).copy()
        flat = (
            pl.read_parquet(
                data_dir
                / f"{test_train_prefix}/{planet_id}/AIRS-CH0_calibration/flat.parquet"
            )
            .to_numpy()
            .astype(np.float64)
            .reshape((32, 356))
        ).copy()
        hot = sigma_clip(dark, sigma=5, maxiters=5).mask.copy()
        dead = (
            pl.read_parquet(
                data_dir
                / f"{test_train_prefix}/{planet_id}/AIRS-CH0_calibration/dead.parquet"
            )
            .to_numpy()
            .astype(bool)
            .reshape((32, 356))
        ).copy()

        c_lib.calibration_pipeline_airs(
            signal.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_double(gain),
            ctypes.c_double(offset),
            linear_corr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            dark.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            dt_airs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(time_binning_freq),
            flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            hot.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            dead.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        len_time_signal_airs = int((len(signal) / 2) / time_binning_freq)
        signal_airs = signal[:len_time_signal_airs, :, CUT_INF:CUT_SUP].transpose(
            0, 2, 1
        )

        ## save data
        np.save(output_dir / f"AIRS_clean_{test_train_prefix}_{n}.npy", signal_airs)
        del signal_airs, signal, gain, offset, linear_corr, dark, flat, hot, dead
        gc.collect()

        ### FGS1 ###
        signal = pl.read_parquet(
            data_dir / f"{test_train_prefix}/{planet_id}/FGS1_signal.parquet"
        )
        signal = signal.to_numpy().astype(np.float64).reshape((signal.shape[0], 32, 32))
        signal = signal.copy()
        gain = adc_info["FGS1_adc_gain"].loc[planet_id]
        offset = adc_info["FGS1_adc_offset"].loc[planet_id]

        dt_fgs1 = np.ones(len(signal)) * 0.1
        dt_fgs1[1::2] += 0.1

        linear_corr = (
            pl.read_parquet(
                data_dir
                / f"{test_train_prefix}/{planet_id}/FGS1_calibration/linear_corr.parquet"
            )
            .to_numpy()
            .astype(np.float64)
            .reshape((6, 32, 32))
        ).copy()
        dark = (
            pl.read_parquet(
                data_dir
                / f"{test_train_prefix}/{planet_id}/FGS1_calibration/dark.parquet"
            )
            .to_numpy()
            .astype(np.float64)
            .reshape((32, 32))
        ).copy()
        flat = (
            pl.read_parquet(
                data_dir
                / f"{test_train_prefix}/{planet_id}/FGS1_calibration/flat.parquet"
            )
            .to_numpy()
            .astype(np.float64)
            .reshape((32, 32))
        ).copy()
        hot = sigma_clip(dark, sigma=5, maxiters=5).mask.copy()
        dead = (
            pl.read_parquet(
                data_dir
                / f"{test_train_prefix}/{planet_id}/FGS1_calibration/dead.parquet"
            )
            .to_numpy()
            .astype(bool)
            .reshape((32, 32))
        ).copy()
        c_lib.calibration_pipeline_fgs1(
            signal.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_double(gain),
            ctypes.c_double(offset),
            linear_corr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            dark.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            dt_fgs1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(time_binning_freq * 12),
            flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            hot.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            dead.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        len_time_signal_fgs1 = int((len(signal) / 2) / (time_binning_freq * 12))
        signal_fgs1 = signal[:len_time_signal_fgs1, :, :].transpose(0, 2, 1)

        ## save data
        np.save(output_dir / f"FGS1_{test_train_prefix}_{n}.npy", signal_fgs1)
        del signal_fgs1, signal, gain, offset, linear_corr, dark, flat, hot, dead
        gc.collect()
    
    def concatenate_files(self):
        print("Concatenating the files...")

        def load_data(file, nb_files):
            data_file = file.with_name(file.name + "_0.npy")
            data0 = np.load(data_file)
            data_file.unlink()

            data_all = np.zeros(
                (nb_files, data0.shape[0], data0.shape[1], data0.shape[2])
            )
            data_all[0] = data0
            for i in range(1, nb_files):
                data_file = file.with_name(file.name + f"_{i}.npy")
                data_all[i : (i + 1)] = np.load(data_file)
                data_file.unlink()

            return data_all

        data_airs = load_data(
            self.output_dir / f"AIRS_clean_{self.test_train_prefix}", len(self.indices_)
        )
        data_fgs1 = load_data(
            self.output_dir / f"FGS1_{self.test_train_prefix}", len(self.indices_)
        )

        print("Saving the files...")
        np.save(self.output_dir / f"data_{self.test_train_prefix}.npy", data_airs)
        np.save(self.output_dir / f"data_{self.test_train_prefix}_FGS.npy", data_fgs1)
        print("Done!")
        
    @staticmethod
    def get_index(files):
        files_len = len(files)
        indices = np.zeros(files_len // 4, dtype=int)
        for i in range(files_len):
            file = files[i]
            file_name_split = file.name.split("_")
            if (
                file_name_split[0] == "AIRS-CH0"
                and file_name_split[1] == "signal.parquet"
            ):
                indices[i // 4] = int(file.parent.name)
        return np.sort(indices)
