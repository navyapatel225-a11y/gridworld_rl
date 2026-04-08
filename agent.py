class Agent:
    def act(self, email):
        subject = email["subject"].lower()

        if "urgent" in subject or "server" in subject:
            return {
                "category": "urgent",
                "action": "escalate",
                "priority": 1,
                "reason": "Detected urgency keywords"
            }

        elif "offer" in subject or "win" in subject:
            return {
                "category": "spam",
                "action": "ignore",
                "priority": 3,
                "reason": "Looks like promotional content"
            }

        else:
            return {
                "category": "normal",
                "action": "reply",
                "priority": 2,
                "reason": "General communication"
            }
