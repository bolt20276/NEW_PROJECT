# import imaplib
# import email
# import datetime
# from email.header import decode_header
# import socket
# import socks

# class EmailFetchError(Exception):
#     """Custom exception for email fetching errors"""
#     pass

# class EmailHandler:
#     def __init__(self, email_user, email_pass):
#         # Corporate network override
#         # Define a placeholder Config class if not already imported
#         class Config:
#             USE_CORPORATE_NETWORK = False  # Set to True if corporate network is used

#         if Config.USE_CORPORATE_NETWORK:
#             try:
#                 socks.set_default_proxy(socks.SOCKS5, "corporate.proxy.com", 1080)
#                 socket.socket = socks.socksocket
#             except Exception as proxy_error:
#                 raise EmailFetchError(f"Proxy connection failed: {str(proxy_error)}")

#         # # DNS override for Yahoo IMAP
#         # if "yahoo" in email_user.lower():
#         #     original_getaddrinfo = socket.getaddrinfo
#         #     def yahoo_getaddrinfo(*args, **kwargs):
#         #         if args[0] == "imap.mail.yahoo.com":
#         #             return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('98.136.96.91', 993))]
#         #         return original_getaddrinfo(*args, **kwargs)
#         #     socket.getaddrinfo = yahoo_getaddrinfo

#         # Rest of initialization
#         self.server = self._detect_server(email_user)
#         self.mail = imaplib.IMAP4_SSL(self.server, 993, timeout=30)

#     def _detect_server(self, email):
#         domain = email.split('@')[-1].lower()
#         server_map = {
#             'gmail.com': 'imap.gmail.com',
#             'yahoo.com': 'imap.mail.yahoo.com',
#             'outlook.com': 'outlook.office365.com',
#             'hotmail.com': 'outlook.office365.com'
#         }
#         return server_map.get(domain, f'imap.{domain}')
    
#     def _decode_subject(self, subject):
#         decoded = decode_header(subject)
#         return ''.join(
#             [part.decode(enc or 'utf-8') if isinstance(part, bytes) else part 
#              for part, enc in decoded]
#         )

#     def _decode_filename(self, filename):
#         decoded = decode_header(filename)
#         return ''.join(
#             [part.decode(enc or 'utf-8') if isinstance(part, bytes) else part 
#              for part, enc in decoded]
#         )

#     def fetch_attachments(self, start_date: str, end_date: str, keywords: list = None, 
#                           max_emails: int = 50, allowed_types: list = ['pdf', 'docx']):
#         """Fetch email attachments with date range and keyword filters"""
#         try:
#             self.mail.select("inbox")
            
#             # Build valid IMAP search query
#             date_query = f'(SINCE "{start_date}" BEFORE "{end_date}")'
#             keyword_query = ""
            
#             if keywords:
#                 # Format: OR (SUBJECT "kw1") (SUBJECT "kw2")
#                 keyword_conditions = ' '.join([f'(SUBJECT "{k}")' for k in keywords])
#                 keyword_query = f'OR {keyword_conditions}' if len(keywords) > 1 else f'SUBJECT "{keywords[0]}"'
#                 keyword_query = f'({keyword_query}) '
            
#             # Combine queries safely
#             search_query = f'{date_query} {keyword_query}'.strip()
            
#             # Use proper IMAP encoding
#             status, messages = self.mail.search('UTF-8', search_query)
            
#             if status != 'OK':
#                 raise EmailFetchError(f"IMAP search failed: {messages[0].decode()}")
                
#             email_ids = messages[0].split()[-max_emails:]
            
#             attachments = []
#             for email_id in email_ids[::-1]:
#                 typ, msg_data = self.mail.fetch(email_id, '(RFC822)')
#                 if typ != 'OK':
#                     continue
                    
#                 msg = email.message_from_bytes(msg_data[0][1])
#                 subject = self._decode_subject(msg.get("Subject", ""))
                
#                 for part in msg.walk():
#                     if part.get_content_maintype() == 'multipart':
#                         continue
                        
#                     filename = part.get_filename()
#                     if filename:
#                         filename = self._decode_filename(filename)
#                         ext = filename.split('.')[-1].lower()
                        
#                         if ext in allowed_types:
#                             content = part.get_payload(decode=True)
#                             attachments.append({
#                                 "filename": filename,
#                                 "content": content,
#                                 "subject": subject,
#                                 "from": msg.get("From", ""),
#                                 "date": msg.get("Date", "")
#                             })
                            
#                 if len(attachments) >= max_emails:
#                     break
                    
#             return attachments
            
#         except Exception as e:
#             raise EmailFetchError(f"Email error: {str(e)}")
#         finally:
#             try:
#                 self.mail.close()
#             except:
#                 pass
#             try:
#                 self.mail.logout()
#             except:
#                 pass


import imaplib
import email
import datetime
from email.header import decode_header
import socket
import socks

class EmailFetchError(Exception):
    """Custom exception for email fetching errors"""
    pass

class EmailHandler:
    def __init__(self, email_user, email_pass):
        # Placeholder Config (adjust or import as needed)
        class Config:
            USE_CORPORATE_NETWORK = False  # Set to True if corporate network is used

        # Corporate proxy handling
        if Config.USE_CORPORATE_NETWORK:
            try:
                socks.set_default_proxy(socks.SOCKS5, "corporate.proxy.com", 1080)
                socket.socket = socks.socksocket
            except Exception as proxy_error:
                raise EmailFetchError(f"Proxy connection failed: {str(proxy_error)}")

        self.server = self._detect_server(email_user)

        try:
            # First try secure connection
            self.mail = imaplib.IMAP4_SSL(self.server, 993, timeout=30)
            self.mail.login(email_user, email_pass)
        except (imaplib.IMAP4.error, TimeoutError, Exception):
            try:
                # Fallback to STARTTLS
                self.mail = imaplib.IMAP4(self.server, 143, timeout=30)
                self.mail.starttls()
                self.mail.login(email_user, email_pass)
            except Exception as fallback_error:
                raise EmailFetchError(f"Email connection failed: {str(fallback_error)}")

    def _detect_server(self, email):
        domain = email.split('@')[-1].lower()
        server_map = {
            'gmail.com': 'imap.gmail.com',
            'yahoo.com': 'imap.mail.yahoo.com',
            'outlook.com': 'outlook.office365.com',
            'hotmail.com': 'outlook.office365.com'
        }
        return server_map.get(domain, f'imap.{domain}')
    
    def _decode_subject(self, subject):
        decoded = decode_header(subject)
        return ''.join(
            [part.decode(enc or 'utf-8') if isinstance(part, bytes) else part 
             for part, enc in decoded]
        )

    def _decode_filename(self, filename):
        decoded = decode_header(filename)
        return ''.join(
            [part.decode(enc or 'utf-8') if isinstance(part, bytes) else part 
             for part, enc in decoded]
        )

    def fetch_attachments(self, start_date: str, end_date: str, keywords: list = None, 
                          max_emails: int = 50, allowed_types: list = ['pdf', 'docx']):
        """Fetch email attachments with date range and keyword filters"""
        try:
            self.mail.select("inbox")
            
            date_query = f'(SINCE "{start_date}" BEFORE "{end_date}")'
            keyword_query = ""

            if keywords:
                keyword_conditions = ' '.join([f'(SUBJECT "{k}")' for k in keywords])
                keyword_query = f'OR {keyword_conditions}' if len(keywords) > 1 else f'SUBJECT "{keywords[0]}"'
                keyword_query = f'({keyword_query}) '

            search_query = f'{date_query} {keyword_query}'.strip()
            status, messages = self.mail.search('UTF-8', search_query)
            
            if status != 'OK':
                raise EmailFetchError(f"IMAP search failed: {messages[0].decode()}")

            email_ids = messages[0].split()[-max_emails:]
            attachments = []

            for email_id in email_ids[::-1]:
                typ, msg_data = self.mail.fetch(email_id, '(RFC822)')
                if typ != 'OK':
                    continue

                msg = email.message_from_bytes(msg_data[0][1])
                subject = self._decode_subject(msg.get("Subject", ""))

                for part in msg.walk():
                    if part.get_content_maintype() == 'multipart':
                        continue

                    filename = part.get_filename()
                    if filename:
                        filename = self._decode_filename(filename)
                        ext = filename.split('.')[-1].lower()

                        if ext in allowed_types:
                            content = part.get_payload(decode=True)
                            attachments.append({
                                "filename": filename,
                                "content": content,
                                "subject": subject,
                                "from": msg.get("From", ""),
                                "date": msg.get("Date", "")
                            })

                if len(attachments) >= max_emails:
                    break

            return attachments

        except Exception as e:
            raise EmailFetchError(f"Email error: {str(e)}")
        finally:
            try:
                self.mail.close()
            except:
                pass
            try:
                self.mail.logout()
            except:
                pass
