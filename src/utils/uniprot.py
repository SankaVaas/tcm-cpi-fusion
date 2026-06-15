"""Fetch protein FASTA sequence from UniProt REST API (no key required)."""
import requests


def fetch_sequence_uniprot(uniprot_id: str, timeout: int = 30) -> str | None:
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    r   = requests.get(url, timeout=timeout)
    if r.status_code != 200:
        return None
    lines = r.text.strip().split("\n")
    return "".join(lines[1:])   # strip FASTA header
