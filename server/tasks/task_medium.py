# server/tasks/task_medium.py
MEDIUM_TASK = {
    "task_id": "task_medium",
    "difficulty": "medium",
    "description": (
        "A corporate HR policy set is missing rules for several emerging workplace "
        "scenarios involving AI tools, remote work, and gig workers. "
        "Identify ONE genuine policy gap and propose a specific new rule to address it."
    ),
    "data_corpus": [
        {"id": "incident_001", "type": "AI_use", "desc": "Employee used ChatGPT to write client proposal without disclosure"},
        {"id": "incident_002", "type": "remote_work", "desc": "Employee attended video call from a coffee shop, client data visible on screen"},
        {"id": "incident_003", "type": "gig_worker", "desc": "Contractor accessed proprietary codebase after project ended"},
        {"id": "incident_004", "type": "AI_use", "desc": "Manager used AI to generate performance review for employee"},
        {"id": "incident_005", "type": "remote_work", "desc": "Employee shared screen showing salary data while on public WiFi"},
        {"id": "incident_006", "type": "gig_worker", "desc": "Freelancer posted client project on portfolio without permission"},
        {"id": "incident_007", "type": "AI_use", "desc": "Employee submitted AI-written code as their own in performance evaluation"},
        {"id": "incident_008", "type": "remote_work", "desc": "Employee worked from another country for 3 months without HR approval"},
        {"id": "incident_009", "type": "gig_worker", "desc": "Contractor attended team standup but was also working for a direct competitor"},
        {"id": "incident_010", "type": "AI_use", "desc": "HR used AI tool to screen resumes — potential bias concerns raised"},
    ],
    "current_policies": [
        {"id": "pol_hr_001", "text": "Employees must maintain confidentiality of client information at all times."},
        {"id": "pol_hr_002", "text": "All employees are expected to comply with the company code of conduct."},
        {"id": "pol_hr_003", "text": "Contractors must sign an NDA before beginning any project."},
        {"id": "pol_hr_004", "text": "Employees working remotely must have a secure, dedicated workspace."},
        {"id": "pol_hr_005", "text": "Any intellectual property created during employment belongs to the company."},
    ],
    "uncovered_domains": ["AI_use", "gig_worker_post_engagement", "cross_border_remote"],
    "num_policies": 5,
    "num_data_points": 10,
}
