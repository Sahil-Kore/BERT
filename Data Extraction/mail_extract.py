import mailbox
from bs4 import BeautifulSoup
import re

# ── CONFIGURE HERE ────────────────────────────────────────────────
MBOX_PATH = "../takeout-20250726T213441Z-1-001/Takeout/Mail/Spam.mbox"
# ─────────────────────────────────────────────────────────────────
import os
import mailbox
from bs4 import BeautifulSoup
import re
import os


def extract_plain_bodies(mbox_path: str, output_dir: str):
    print(f"Processing MBOX file: {mbox_path}")
    os.makedirs(output_dir, exist_ok=True)

    mbox = mailbox.mbox(mbox_path)

    # Compile regex patterns once
    patterns = {
        "url": re.compile(r'(https?://\S+|www\.\S+)'),
        "tpl_block": re.compile(r'{%.*?%}', re.DOTALL),
        "tpl_var": re.compile(r'{{[^}]*}}'),
        "empty_paren": re.compile(r'\(\s*\)'),
        "lid_paren": re.compile(r'\(\s*[?&]lid=[^)]+\)'),
        "css_comment": re.compile(r'/\*[\s\S]*?\*/', re.DOTALL),
        "css_rule": re.compile(r'[^\{]*\{[^}]*\}', re.DOTALL),
        "html_tag": re.compile(r'<[^>]+>'),
        # Remove zero-width and non-breaking spaces
        "zero_width": re.compile(r'[\u200B\u200C\u200D\uFEFF]'),
        "nbsp": re.compile(r'\u00A0'),
    }

    def clean_text(body: str) -> str:
        # Normalize unicode spaces
        body = body.replace('\u00A0', ' ')

        # Remove unwanted patterns
        for pattern in patterns.values():
            body = pattern.sub("", body)

        # Line-by-line filter cleanup
        lines = []
        for line in body.splitlines():
            stripped = line.strip()
            # Skip CSS-like lines and selectors
            if (';' in stripped and ':' in stripped) or stripped.startswith('@') or \
               '{' in stripped or '}' in stripped:
                continue
            if ',' in stripped and ':' not in stripped and ';' not in stripped:
                continue
            lines.append(line)
        body = "\n".join(lines)

        # Collapse multiple spaces/tabs and other whitespace to single space
        body = re.sub(r'[ \t\x0b\x0c]+', ' ', body)
        # Collapse multiple blank lines into a single blank line
        body = re.sub(r'\n\s*\n+', '\n\n', body)
        # Trim leading/trailing whitespace and newlines
        return body.strip()

    def get_email_body(msg) -> str:
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                dispo = str(part.get("Content-Disposition") or "")
                if "attachment" in dispo:
                    continue
                payload = part.get_payload(decode=True) or b""
                text = payload.decode(errors="ignore")
                if ctype == "text/plain":
                    return text
                elif ctype == "text/html":
                    return BeautifulSoup(text, "html.parser").get_text()
        else:
            payload = msg.get_payload(decode=True) or b""
            text = payload.decode(errors="ignore")
            if msg.get_content_type() == "text/plain":
                return text
            elif msg.get_content_type() == "text/html":
                return BeautifulSoup(text, "html.parser").get_text()
        return ""

    for count, msg in enumerate(mbox, 1):
        raw_body = get_email_body(msg)
        if not raw_body:
            continue

        clean_body = clean_text(raw_body)
        if clean_body:
            save_path = os.path.join(output_dir, f"{output_dir}_{count:05}.txt")
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(clean_body)
                

def delete_short_files(folder_path: str, min_chars: int = 15):
    count=0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if not filename.endswith(".txt") or not os.path.isfile(file_path):
            continue

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        if len(content.strip()) < min_chars:
            print(f"Deleting {filename} (length: {len(content.strip())})")
            os.remove(file_path)
            count+=1
    print(count)
if __name__ == "__main__":
    extract_plain_bodies(MBOX_PATH,"Spam")
    delete_short_files("./Inbox")

