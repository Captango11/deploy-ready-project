"""Local NLP skill dictionary and skill extraction (regex over normalized text)."""
import re
from .preprocessing import normalize

# canonical skill -> patterns matched against *normalized* text
SKILL_PATTERNS = {
    "Python": [r"\bpython\b"],
    "Java": [r"\bjava\b"],
    "C++": [r"\bcplusplus\b"],
    "C#": [r"\bcsharp\b"],
    "JavaScript": [r"\bjavascript\b", r"\bjs\b"],
    "TypeScript": [r"\btypescript\b"],
    "SQL": [r"\bsql\b"],
    "MySQL": [r"\bmysql\b"],
    "PostgreSQL": [r"\bpostgresql\b", r"\bpostgres\b"],
    "MongoDB": [r"\bmongodb\b", r"\bmongo\b"],
    "React": [r"\breact\b"],
    "Node.js": [r"\bnodejs\b"],
    "Express": [r"\bexpress\b"],
    "HTML": [r"\bhtml5?\b"],
    "CSS": [r"\bcss3?\b"],
    "Tailwind": [r"\btailwind\b"],
    "Pandas": [r"\bpandas\b"],
    "NumPy": [r"\bnumpy\b"],
    "Scikit-learn": [r"\bscikit_learn\b"],
    "TensorFlow": [r"\btensorflow\b"],
    "PyTorch": [r"\bpytorch\b"],
    "Machine Learning": [r"\bmachine learning\b", r"\bml models?\b"],
    "Deep Learning": [r"\bdeep learning\b", r"\bneural networks?\b"],
    "NLP": [r"\bnlp\b", r"\bnatural language processing\b"],
    "Computer Vision": [r"\bcomputer vision\b", r"\bopencv\b"],
    "LLMs": [r"\bllms?\b", r"\blarge language models?\b"],
    "Transformers": [r"\btransformers?\b"],
    "RAG": [r"\brag\b", r"\bretrieval augmented\b"],
    "LangChain": [r"\blangchain\b"],
    "Vector Databases": [r"\bvector (databases?|search|db)\b", r"\bpinecone\b", r"\bfaiss\b"],
    "Prompt Engineering": [r"\bprompt engineering\b"],
    "MLOps": [r"\bmlops\b", r"\bmlflow\b"],
    "Feature Engineering": [r"\bfeature engineering\b"],
    "Power BI": [r"\bpower_bi\b"],
    "Tableau": [r"\btableau\b"],
    "Looker": [r"\blooker\b"],
    "Excel": [r"\bexcel\b"],
    "AWS": [r"\baws\b", r"\bamazon web services\b"],
    "Azure": [r"\bazure\b"],
    "GCP": [r"\bgcp\b", r"\bgoogle cloud\b"],
    "Lambda": [r"\blambda\b"],
    "S3": [r"\bs3\b"],
    "Docker": [r"\bdocker\b"],
    "Kubernetes": [r"\bkubernetes\b", r"\bk8s\b"],
    "Linux": [r"\blinux\b"],
    "Bash": [r"\bbash\b", r"\bshell scripting\b"],
    "Terraform": [r"\bterraform\b"],
    "Ansible": [r"\bansible\b"],
    "Git": [r"\bgit\b", r"\bgithub\b", r"\bgitlab\b"],
    "FastAPI": [r"\bfastapi\b"],
    "Flask": [r"\bflask\b"],
    "Django": [r"\bdjango\b"],
    "Spring Boot": [r"\bspring_boot\b"],
    "REST API": [r"\brest_api\b", r"\bapis?\b"],
    "Microservices": [r"\bmicroservices?\b"],
    "Redis": [r"\bredis\b"],
    "Kafka": [r"\bkafka\b"],
    "Spark": [r"\bspark\b", r"\bpyspark\b"],
    "Hadoop": [r"\bhadoop\b"],
    "ETL": [r"\betl\b"],
    "Statistics": [r"\bstatistics\b", r"\bstatistical\b"],
    "A/B Testing": [r"\bab_testing\b"],
    "Data Visualization": [r"\bdata visualization\b", r"\bdashboards?\b"],
    "Jenkins": [r"\bjenkins\b"],
    "GitHub Actions": [r"\bgithub_actions\b"],
    "CI/CD": [r"\bcicd\b"],
    "Prometheus": [r"\bprometheus\b", r"\bgrafana\b"],
    "Monitoring": [r"\bmonitoring\b", r"\bobservability\b"],
    "Unit Testing": [r"\bunit test(ing|s)?\b", r"\bjunit\b", r"\bpytest\b"],
    "OOP": [r"\boop\b", r"\bobject oriented\b"],
    "Data Structures": [r"\bdata structures?\b", r"\balgorithms?\b"],
    "Design Patterns": [r"\bdesign patterns?\b"],
    "Agile": [r"\bagile\b", r"\bscrum\b", r"\bjira\b"],
    "Requirements Gathering": [r"\brequirements? (gathering|analysis)\b", r"\brequirements\b"],
    "Stakeholder Management": [r"\bstakeholders?\b"],
    "Process Modeling": [r"\bbpmn\b", r"\bprocess (modeling|mapping|improvement)\b"],
    "Business Analysis": [r"\bbusiness analysis\b", r"\bgap analysis\b"],
    "Networking": [r"\bnetworking\b", r"\bvpcs?\b"],
    "Cost Optimization": [r"\bcost optimization\b"],
}

ALL_SKILLS = list(SKILL_PATTERNS.keys())
_COMPILED = {s: [re.compile(p) for p in ps] for s, ps in SKILL_PATTERNS.items()}


def extract_skills(raw_text: str) -> list:
    """Detect skills that literally appear in the resume text."""
    norm = normalize(raw_text)
    return [s for s, pats in _COMPILED.items() if any(p.search(norm) for p in pats)]


def skill_in_text(skill: str, raw_text: str) -> bool:
    norm = normalize(raw_text)
    return any(p.search(norm) for p in _COMPILED.get(skill, []))
