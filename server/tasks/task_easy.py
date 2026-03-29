# server/tasks/task_easy.py
EASY_TASK = {
    "task_id": "task_easy",
    "difficulty": "easy",
    "description": (
        "A social media platform has vague community guidelines. "
        "Identify ONE genuinely ambiguous term and propose a specific, "
        "measurable definition that moderators can apply consistently."
    ),
    "data_corpus": [
        {"id": "post_001", "text": "You're so ugly, go back to where you came from", "action_taken": "removed"},
        {"id": "post_002", "text": "I hate Mondays so much", "action_taken": "allowed"},
        {"id": "post_003", "text": "This politician is a clown and should resign", "action_taken": "allowed"},
        {"id": "post_004", "text": "Women shouldn't be in leadership roles", "action_taken": "removed"},
        {"id": "post_005", "text": "I can't stand my neighbor's dog barking", "action_taken": "allowed"},
        {"id": "post_006", "text": "People like you ruin everything", "action_taken": "flagged"},
        {"id": "post_007", "text": "Go to hell with your stupid opinions", "action_taken": "flagged"},
        {"id": "post_008", "text": "This food is disgusting, whoever made it should be ashamed", "action_taken": "allowed"},
    ],
    "current_policies": [
        {"id": "pol_001", "text": "Users must not post content that is offensive or inappropriate."},
        {"id": "pol_002", "text": "Harassment of any kind is strictly prohibited."},
        {"id": "pol_003", "text": "Content that promotes hate speech will be removed."},
        {"id": "pol_004", "text": "Users should communicate in a respectful manner."},
    ],
    "known_ambiguous_terms": ["offensive", "inappropriate", "harassment", "hate speech", "respectful"],
    "num_policies": 4,
    "num_data_points": 8,
}
