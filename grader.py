def grade(email, action):
    subject = email["subject"].lower()

    expected = "normal"
    if "urgent" in subject or "server" in subject:
        expected = "urgent"
    elif "offer" in subject or "win" in subject:
        expected = "spam"

    score = 0

    if action["category"] == expected:
        score += 1

    if "reason" in action:
        score += 0.5

    return score
