import os
import pandas as pd

# =============================
# Config
# =============================
OUTPUT_DIR = "overlap_results"
TIMELINE_CSV = os.path.join(OUTPUT_DIR, "speaker_segments.csv")
OVERLAP_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "overlap_summary.csv")


def calculate_audio_overlap(df_audio):
    events = []
    for _, row in df_audio.iterrows():
        events.append((row["start"], 1))
        events.append((row["end"], -1))

    events.sort(key=lambda x: (x[0], x[1]))

    active_duration = 0.0
    overlap_duration = 0.0
    current_speakers = 0
    prev_time = 0.0

    for time, change in events:
        dt = time - prev_time

        # 至少有 1 个人在说话，计入有效语音时长
        if current_speakers >= 1:
            active_duration += dt

        # 至少有 2 个人在同时说话，计入重叠语音时长
        if current_speakers >= 2:
            overlap_duration += dt

        current_speakers += change
        prev_time = time

    return active_duration, overlap_duration


def main():
    if not os.path.exists(TIMELINE_CSV):
        print(f"错误: 找不到输入文件 {TIMELINE_CSV}")
        return

    df = pd.read_csv(TIMELINE_CSV)

    summary_records = []

    # 按音频文件名分组处理
    for audio_name, group in df.groupby("audio"):
        active_dur, overlap_dur = calculate_audio_overlap(group)

        # 音频中最后一个说话人结束的时间（近似音频长度）
        max_end_time = group["end"].max()

        # 各说话人单人片段时长直接相加（未去重）
        total_speech_sum = group["duration"].sum()

        # 计算两种重叠比例 (保留 4 位小数)
        overlap_ratio_active = (
            round(overlap_dur / active_dur, 4) if active_dur > 0 else 0.0
        )
        overlap_ratio_total = (
            round(overlap_dur / max_end_time, 4) if max_end_time > 0 else 0.0
        )

        summary_records.append(
            {
                "audio": audio_name,
                "active_speech_duration": round(active_dur, 3),
                "overlap_duration": round(overlap_dur, 3),
                "max_end_time": round(max_end_time, 3),
                "total_speaker_sum_duration": round(total_speech_sum, 3),
                "overlap_ratio_active": overlap_ratio_active,  # 基于有效语音时长的重叠率
                "overlap_ratio_total": overlap_ratio_total,  # 基于最大结束时间的重叠率
            }
        )

    summary_df = pd.DataFrame(summary_records)

    summary_df.to_csv(OVERLAP_SUMMARY_CSV, index=False)


    print(summary_df.head())



if __name__ == "__main__":
    main()