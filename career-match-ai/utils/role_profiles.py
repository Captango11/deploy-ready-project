"""Project-defined role-skill profiles. To add a role: add training rows to
data/resumes.csv, add an entry here, and re-run train_model.py."""

ROLE_SKILLS = {
    "Data Analyst": ["Python", "SQL", "Excel", "Power BI", "Tableau", "Pandas",
                     "Statistics", "Data Visualization", "Looker", "A/B Testing"],
    "Data Scientist": ["Python", "SQL", "Statistics", "Machine Learning", "Pandas",
                       "NumPy", "Scikit-learn", "Feature Engineering", "Data Visualization",
                       "A/B Testing"],
    "ML Engineer": ["Python", "Machine Learning", "Scikit-learn", "TensorFlow", "PyTorch",
                    "Docker", "REST API", "MLOps", "Kubernetes", "FastAPI"],
    "AI Engineer": ["Python", "Deep Learning", "NLP", "LLMs", "Transformers", "PyTorch",
                    "RAG", "Vector Databases", "LangChain", "Prompt Engineering"],
    "Web Developer": ["HTML", "CSS", "JavaScript", "TypeScript", "React", "Node.js",
                      "Tailwind", "REST API", "Git"],
    "Backend Developer": ["Python", "Java", "REST API", "PostgreSQL", "MySQL", "Docker",
                          "Django", "Spring Boot", "Microservices", "Redis", "Kafka"],
    "Software Developer": ["Java", "C++", "Python", "Git", "OOP", "Data Structures",
                           "Design Patterns", "Unit Testing", "Agile"],
    "Cloud Engineer": ["AWS", "Azure", "GCP", "Terraform", "Lambda", "S3", "Networking",
                       "Linux", "Cost Optimization", "Docker"],
    "DevOps Engineer": ["Docker", "Kubernetes", "Jenkins", "GitHub Actions", "CI/CD",
                        "Terraform", "Linux", "Bash", "Ansible", "Prometheus"],
    "Business Analyst": ["Excel", "SQL", "Requirements Gathering", "Stakeholder Management",
                         "Process Modeling", "Agile", "Business Analysis", "Power BI",
                         "Data Visualization"],
}

ROLES = list(ROLE_SKILLS.keys())
