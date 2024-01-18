import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path

def csv_path(csv):
    path = Path.cwd()
    path = str(path)
    path = path[:-4]
    path = path + csv
    f_path = Path(path)
    return f_path

fromaddr = "saravanashanmuganathan35@gmail.com"
toaddr = ["saravana.shanmuganathan@axtria.com"]

msg = MIMEMultipart()

for i in toaddr:
    msg['From'] = fromaddr
    msg['To'] = i
    msg['Subject'] = "Data Drift detected"
    body = ''' Hi Team, 
        Data Drift has happened please check the data.
    Thank you'''
    msg.attach(MIMEText(body, 'plain'))
    filename = "drift_result.html"
    attachment = open(csv_path("//result//drift_result.html"), "rb")
    p = MIMEBase('application', 'octet-stream')
    p.set_payload((attachment).read())
    encoders.encode_base64(p)
    p.add_header('Content-Disposition', "attachment; filename= %s" % filename)
    msg.attach(p)
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login(fromaddr, "ngld jpvc mszi kejt")
    text = msg.as_string()
    print(text)
    s.sendmail(fromaddr, i, text)
    s.quit()
