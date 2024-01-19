import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path

smtp_server = "smtp.gmail.com"
smtp_port = 587  # TLS port for Gmail
smtp_username = "saravanashanmuganathan35@gmail.com"  # Your Gmail email address
smtp_password = "ngld jpvc mszi kejt"  # Your generated Gmail App Password

def csv_path(csv):
    path = Path.cwd()
    path = str(path)
    path = path[:-4]
    path = path + csv
    f_path = Path(path)
    return f_path
# Define the email sender function
def send_email(subject, body, to_address):
    from_email = smtp_username

    # Create the email message
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_address
    msg['Subject'] = subject

    # Attach the email body
    msg.attach(MIMEText(body, 'plain'))
    filename = "drift_result.html"
    attachment = open(csv_path("//result//drift_result.html"), "rb")
    p = MIMEBase('application', 'octet-stream')
    p.set_payload((attachment).read())
    encoders.encode_base64(p)
    p.add_header('Content-Disposition', "attachment; filename= %s" % filename)
    msg.attach(p)
    text = msg.as_string()

    try:
        # Connect to the Gmail SMTP server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Use TLS for encryption
        server.login(smtp_username, smtp_password)

        # Send the email
        server.sendmail(from_email, to_address, text)
        server.quit()

        return {"message": "Email sent successfully!"}
    except Exception as e:
        print(f"Exception when sending email: {str(e)}")
        return {"message": "Email sending failed."}



content = ''' Hi Team, 
               Data Drift has happened, please check the data. There is an change in data.
Thank you'''
subject = "drift detection found"
to_address = ["saravana.shanmuganathan@axtria.com","vaidhyanathan.v@axtria.com"]

print("Sending mail...")
for i in to_address:
    # Send the email and store the response
    email_response = send_email(subject, content, i)
    # Print the status of the email sending process
    print(email_response)
    print("mail sent to "+i)
