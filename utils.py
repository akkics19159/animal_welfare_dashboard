def calculate_welfare_score(video_flags, audio_flags, sensor_flags):
    score = 100

    if audio_flags.get("distress"):
        score -= 35
    if video_flags.get("motion_agitation"):
        score -= 25
    if video_flags.get("motion_score", 0.0) > 0.20:
        score -= 10
    if sensor_flags.get("temp_high"):
        score -= 12
    if sensor_flags.get("humidity_extreme"):
        score -= 10
    if sensor_flags.get("heart_rate_extreme"):
        score -= 20
    if audio_flags.get("distress") and not video_flags.get("sentient_present"):
        score -= 10

    return max(score, 0)


def explain_flags(video_flags, audio_flags, sensor_flags):
    reasons = []
    if video_flags.get("missing"):
        reasons.append("Video input unavailable or unreliable")
    if audio_flags.get("missing"):
        reasons.append("Audio input unavailable or unreliable")
    if sensor_flags.get("missing"):
        reasons.append("Sensor input unavailable or unreliable")
    if video_flags.get("sentient_present"):
        count = video_flags.get("sentient_count", 0)
        reasons.append(f"Detected {count} sentient being(s) in video feed")
        if video_flags.get("motion_agitation"):
            reasons.append("Detected agitated or pacing movement")
    if audio_flags.get("distress"):
        reasons.append("Detected distress-like audio patterns")
        reasons.append(f"Audio distress score: {audio_flags.get('score', 0.0):.2f}")
    if sensor_flags.get("temp_high"):
        reasons.append("Temperature above safe threshold")
    if sensor_flags.get("humidity_extreme"):
        reasons.append("Humidity outside comfortable range")
    if sensor_flags.get("heart_rate_extreme"):
        reasons.append("Heart rate outside expected range")
    if audio_flags.get("distress") and not video_flags.get("sentient_present"):
        reasons.append("Audio distress detected without visible beings")
    if not reasons:
        reasons.append("No clear suffering indicators detected")
    return reasons
