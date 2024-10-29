import email
import mailbox
the_mailbox = mailbox.mbox(r"C:\Users\Rory\OneDrive - University of Canterbury\Desktop\Code\TackTech\archive\All mail Including Spam and Trash-002 (1).mbox")

def get_body(message: email.message.Message, encoding: str = "utf-8") -> str:
    body_in_bytes = ""
    if message.is_multipart():
        for part in message.walk():
            ctype = part.get_content_type()
            cdispo = str(part.get("Content-Disposition"))

            if ctype == "text/plain" and "attachment" not in cdispo:
                body_in_bytes = part.get_payload(decode=True) 
                break
    else:
        body_in_bytes = message.get_payload(decode=True)

    body = body_in_bytes.decode(encoding)

    return body

for message in the_mailbox:
    content = get_body(message)
    break