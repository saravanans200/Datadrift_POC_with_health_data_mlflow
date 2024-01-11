import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Email configuration
sender_email = "saravanashanmuganathan35@gmail.com"
receiver_email = "saravana.shanmuganathan@axtria.com"
password = "saravana*35"

# Create the email message
subject = "Subject of the email"
body = "Body of the email"
message = MIMEMultipart()
message['From'] = sender_email
message['To'] = receiver_email
message['Subject'] = subject
message.attach(MIMEText(body, 'plain'))

# Establish a connection to the SMTP server
smtp_server = "smtp.gmail.com"
smtp_port = 587
with smtplib.SMTP(smtp_server, smtp_port) as server:
    server.starttls()  # Use TLS encryption
    server.login(sender_email, password)
    server.sendmail(sender_email, receiver_email, message.as_string())

print("Email sent successfully!")
