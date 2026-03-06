# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyroomacoustics",
#     "soundfile",
#     "numpy",
# ]
# ///

import argparse

import numpy as np
import pyroomacoustics as pra
import soundfile as sf


NUM_MICS = 16
DIAMETER_M = 0.4


def get_mic_array():
    radius = DIAMETER_M / 2
    angles = np.linspace(0, 2 * np.pi, NUM_MICS, endpoint=False)
    coords = np.zeros((NUM_MICS, 3))
    coords[:, 0] = radius * np.cos(angles)
    coords[:, 1] = radius * np.sin(angles)
    return coords


def simulate_array(input_path: str, output_path: str, azimuth_deg: float = 45.0,
                   distance_m: float = 5.0):
    audio, fs = sf.read(input_path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float32)

    az_rad = np.radians(azimuth_deg)
    source_offset = np.array([
        distance_m * np.sin(az_rad),
        distance_m * np.cos(az_rad),
        0.0,
    ])

    room_dim = [15.0, 15.0, 10.0]
    array_center = np.array(room_dim) / 2.0
    source_pos = np.clip(array_center + source_offset, 0.5, np.array(room_dim) - 0.5)

    mic_coords_relative = get_mic_array()
    mic_coords = mic_coords_relative + array_center

    room = pra.ShoeBox(
        room_dim,
        fs=fs,
        materials=pra.Material(0.9, 0.1),
        max_order=1,
    )
    room.add_source(source_pos, signal=audio)
    room.add_microphone_array(mic_coords.T)

    room.compute_rir()
    room.simulate()

    output = room.mic_array.signals.T
    sf.write(output_path, output, fs)
    print(f"Wrote {output.shape[1]}-channel output to {output_path}")


def estimate_doa(input_path: str) -> float:
    audio, fs = sf.read(input_path)
    if audio.ndim == 1:
        raise ValueError("Input must be multi-channel audio")

    mic_coords = get_mic_array()
    nfft = 256
    freq_range = [300, 3500]

    # Compute STFT - locate_sources expects frequency domain data
    X = pra.transform.stft.analysis(audio, nfft, nfft // 2)
    X = X.transpose([2, 1, 0])  # Shape: (n_freq, n_channels, n_frames)

    doa = pra.doa.SRP(mic_coords.T, fs, nfft, c=343.0, num_src=1,
                      mode="far", azimuth=np.linspace(-np.pi, np.pi, 360))
    doa.locate_sources(X, freq_range=freq_range)

    estimated_az = np.degrees(doa.azimuth_recon[0])
    print(f"Estimated azimuth: {estimated_az:.1f}°")

    return estimated_az


def main():
    parser = argparse.ArgumentParser(description="Simulate 16-channel microphone array from mono WAV")
    parser.add_argument("input", help="Input mono WAV file")
    parser.add_argument("--output", "-o", default="output_16ch.wav", help="Output 16-channel WAV file")
    parser.add_argument("--azimuth", type=float, default=45.0, help="Source azimuth in degrees")
    parser.add_argument("--distance", type=float, default=5.0, help="Source distance in meters")
    args = parser.parse_args()

    simulate_array(args.input, args.output, args.azimuth, args.distance)

    print(f"\nGround truth: azimuth={args.azimuth}°")
    estimate_doa(args.output)


if __name__ == "__main__":
    main()
