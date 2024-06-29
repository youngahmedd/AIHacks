from urllib.request import urlopen
import html2text
url='https://www.microsoft.com/en-us/investor/earnings/fy-2024-q3/press-release-webcast'
page = urlopen(url)
html_content = ""
if page.headers.get_content_charset():
    html_content = page.read().decode(page.headers.get_content_charset())
else:
    html_content = page.read().decode("utf-8")
rendered_content = html2text.html2text(html_content)
file = open('./txt_files/file_text.txt', 'w')
file.write(rendered_content)
file.close()