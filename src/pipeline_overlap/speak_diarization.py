import os
import pandas as pd
from pyannote.audio import Pipeline

# =============================
# Config
# =============================

HF_TOKEN = "YOUR_TOKEN"

AUDIO_DIR = "output_audio"

OUTPUT_DIR = "overlap_results"

TIMELINE_CSV = os.path.join(
    OUTPUT_DIR,
    "speaker_segments.csv"
)

# =============================
# Load pipeline
# =============================

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=HF_TOKEN
)

# =============================
# Main
# =============================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    records = []

    audio_files = sorted(os.listdir(AUDIO_DIR))

    print(f"Found {len(audio_files)} audio files.")

    for audio in audio_files:

        if not audio.endswith(".wav"):
            continue

        audio_path = os.path.join(
            AUDIO_DIR,
            audio
        )

        print(f"\nProcessing {audio}")

        diarization = pipeline(audio_path)

        for turn, speaker in diarization.speaker_diarization:

            records.append({

                "audio": audio,

                "speaker": speaker,

                "start": round(turn.start,3),

                "end": round(turn.end,3),

                "duration": round(
                    turn.end-turn.start,
                    3
                )

            })

    df = pd.DataFrame(records)

    df.to_csv(
        TIMELINE_CSV,
        index=False
    )

    print(df.head())

    print()

    print("Saved:")

    print(TIMELINE_CSV)


if __name__=="__main__":

    main()