import csv

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from bs4 import BeautifulSoup
import pandas as pd
import requests
import re
import logging
import configparser
import time
import openai
import requests
from pdfminer.high_level import extract_text


# Basic configs:
logging.basicConfig(filename='log.log', level=logging.INFO, format='%(asctime)s %(message)s')
config = configparser.ConfigParser()
config.read('config.ini')
openai.api_key = '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'

# Load PICOC terms
picoc = {
    'population': re.compile(config['picoc']['population']) if config['picoc']['population'] != "" else None,
    'intervention': re.compile(config['picoc']['intervention']) if config['picoc']['intervention'] != "" else None,
    'comparison': re.compile(config['picoc']['comparison']) if config['picoc']['comparison'] != "" else None,
    'outcome': re.compile(config['picoc']['outcome']) if config['picoc']['outcome'] != "" else None,
    'context': re.compile(config['picoc']['context']) if config['picoc']['context'] != "" else None
}

# Create a new Chorme session

options = None
if config['default']['binary_location']:
    options = Options()
    options.binary_location = config['default']['binary_location']

driver = webdriver.Chrome(r"C:\Users\karet\Downloads\chromedriver_win32\chromedriver", options=options)
url = "https://scholar.google.com/"
driver.get(url)

# Setting Google Scholar
driver.maximize_window()
time.sleep(1)
driver.find_element_by_id("gs_hdr_mnu").click()
time.sleep(1)
driver.find_element_by_class_name("gs_btnP").click()
time.sleep(1)
driver.find_element_by_id("gs_num-b").click()
time.sleep(1)
driver.find_element_by_css_selector('a[data-v="20"').click()
time.sleep(1)
driver.find_element_by_id("gs_settings_import_some").click()
time.sleep(1)
driver.find_element_by_name("save").click()


def split_content_into_chunks(content, chunk_size=16000):  # chunk_size can be adjusted as needed
    return [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]


def clean_html_content(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract just the body content
    body = soup.body

    # Remove <script> and <style> tags from the body
    [s.extract() for s in body(['script', 'style'])]

    # Convert back to string and return
    return str(body)


def extract_text_from_pdf(pdf_link):
    try:
        # Download PDF
        response = requests.get(pdf_link)
        response.raise_for_status()  # Check if the request was successful

        file_path = "temp.pdf"
        with open(file_path, "wb") as f:
            f.write(response.content)

        # Extract text from the first two pages
        return extract_text(file_path, page_numbers=[0, 1,2,3,4])
    except requests.RequestException as e:
        print(f"Error downloading PDF: {e}")
        return ""
    except Exception as e:
        print(f"Error occurred while processing the PDF: {e}")
        return ""


def analyze_abstract(abstract):
    # Step 1: Check if the abstract discusses a simulation with LLM models
    response_1 = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "user",
             "content": "Does this abstract describe a specific simulation created by the authors using LLM models? Please give one word as an Yes/No answer. Abstract: " + abstract}
        ]
    )
    is_llm_simulation = response_1.choices[0].message['content'].strip().lower() == 'yes'

    if not is_llm_simulation:
        return False, None, None,None

    # Step 2: Identify the type of simulation
    response_2 = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "user", "content": "What type of simulation does this abstract describe? Please give one word as an answer."
                                        "If there is no information to asnwer this questions, state NA"
                                        "Abstract: " + abstract}
        ]
    )
    types = response_2.choices[0].message['content'].strip()

    # Step 3: Outlining the main benefits
    response_3 = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "user",
             "content": "What are the main benefits of the simulation described in this abstract?"
                        "Desribe shortly"
                        "If there is no information to asnwer this questions, state NA."
                        "Abstract: " + abstract}
        ]
    )
    benefits = response_3.choices[0].message['content'].strip()


    # Step 4: Outlining the main application
    response_4 = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "user",
             "content": "What is the primary application of LLM in this social simulation?"
                        "Desribe shortly."
                        "If there is no information to asnwer this questions, state NA."
                        "Abstract: " + abstract}
        ]
    )
    applications = response_4.choices[0].message['content'].strip()


    return is_llm_simulation, types,applications, benefits


import time


def get_abstract_from_chatgpt(content, content_type="html", max_retries=3, initial_delay=5):
    # Preprocess content based on its type
    if content_type == "html":
        cleaned_content = clean_html_content(content)
    elif content_type == "pdf":
        cleaned_content = content  # Assuming PDF content is already text and doesn't need further cleaning
    else:
        raise ValueError(f"Unknown content_type: {content_type}")

    chunks = split_content_into_chunks(cleaned_content)

    for chunk in chunks:
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Construct the role message based on content type
                system_message = f"You are a helpful assistant that extracts abstracts from {content_type.upper()} content."

                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-16k",
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user",
                         "content": f"Extract abstract from the following {content_type.upper()} content: {chunk}. Output only it and nothing else."
                                    f"Hint: usually it starts from the word abstract"
                                    f" If you did not find an abstract, please output 'Not Found'."}
                    ]
                )

                abstract = response.choices[0].message['content'].strip()
                if abstract.lower() != 'not found':
                    return abstract
                break  # Break out of the retry loop if no server error occurred
            except Exception as e:
                if "The server is overloaded or not ready yet" in str(e) and retry_count < max_retries - 1:
                    wait_time = initial_delay * (2 ** retry_count)  # Exponential backoff
                    print(f"Server error detected. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    retry_count += 1
                else:
                    raise  # Raise the exception if it's a different error or if retries are exhausted

    # If all chunks were processed and no abstract was found:
    return "Not Found"


# Sometimes a Captcha shows up. It needs to be fixed manually. This function makes the code wait until this be fixed
def check_captcha():
    captcha = driver.find_elements_by_css_selector("#captcha")
    captcha += driver.find_elements_by_css_selector("#gs_captcha_f")
    captcha += driver.find_elements_by_css_selector("#g-recaptcha")
    captcha += driver.find_elements_by_css_selector("#recaptcha")
    while captcha:
        logging.info(
            "Captcha found! You need to fill it on browser to continue. Go to the terminal and type 'y' when the Captcha be solved")
        print("Captcha found! You need to fill it on browser to continue...")
        solve = input("Type 'y' when the Captcha be solved: ")
        if solve == "y":
            try:
                driver.find_element_by_id("gs_res_sb_yyc")
                logging.info("Captcha solved, continuing...")
                break
            except:
                print("Captcha not solved, try again! You need to fill it on browser to continue...")
                logging.info("Captcha not solved, try again! You need to fill it on browser to continue...")
        else:
            print("Input error. Try again")


# Filter the PICOC terms inside the Title-Abstract-Keywords
def filterTitleAbsKey(site):
    try:
        page = requests.get(site,timeout=600)
        text = BeautifulSoup(page.text, 'lxml').get_text()
        text = str.lower(text)
        for terms in filter(None, picoc.values()):
            if not terms.search(text):
                logging.info("%s not passed on title-abs-key filter", site)
                return False
        logging.info("%s passed on title-abs-key filter", site)
        return True
    except requests.exceptions.Timeout:
        logging.info("[TIMEOUT] Timeout on %s and not passed on title-abs-key filter. Skipping website", site)
    except:
        logging.info("[ERROR] on %s and not passed on title-abs-key filter", site)
    return False


def extract_abstract(content, content_type):
    """Utility function to get abstract and handle errors"""
    try:
        abstract = get_abstract_from_chatgpt(content, content_type=content_type)
        print(abstract)
        return abstract
    except Exception as e:
        print(f"Error occurred: {e}")
        return ''

def is_pdf(link):
    try:
        # Get a small portion of the file (first 5 bytes should suffice for checking the PDF signature)
        response = requests.get(link, stream=True, timeout=5)
        start_of_file = response.content[:5]
        return start_of_file == b'%PDF-'
    except requests.RequestException:
        return False

# Parser HTML
def parser(soup, page, year):
    papers = []
    html = soup.findAll('div', {'class': 'gs_r gs_or gs_scl'})
    for result in html:
        paper = {'Link': result.find('h3', {'class': "gs_rt"}).find('a')['href'], 'Additional link': '', 'Title': '',
                 'Authors': '', 'Abstract': '', 'Cited by': '', 'Cited list': '', 'Related list': '', 'Bibtex': '',
                 'Year': year, 'Google page': page, "Is LLM Simulation":'', "Simulation Type": '',"Application Type":'', "Simulation Benefits":''}

        # If it does not pass at Title-Abstract-Keyword filter exclude this paper and continue
        """
        if not filterTitleAbsKey(paper['Link']):
            continue

        """


        paper['Title'] = result.find('h3', {'class': "gs_rt"}).text
        print(paper['Title'])
        paper['Authors'] = ";".join(
            ["%s:%s" % (a.text, a['href']) for a in result.find('div', {'class': "gs_a"}).findAll('a')])

        # Main logic
        if is_pdf(paper["Link"]):
            pdf_text = extract_text_from_pdf(paper["Link"])
            paper["Abstract"] = extract_abstract(pdf_text, content_type="pdf")
        else:
            driver.get(paper["Link"])
            time.sleep(2)
            papier_page_soup = BeautifulSoup(driver.page_source, 'html.parser')
            html_content = str(papier_page_soup)
            paper["Abstract"] = extract_abstract(html_content, content_type="html")

        # Analyzing the abstract
        is_llm_simulation, types,applications, benefits = analyze_abstract(paper["Abstract"])

        # Storing the results in the paper dictionary
        paper["Is LLM Simulation"] = is_llm_simulation
        paper["Simulation Type"] = types
        paper["Application Type"] = applications
        paper["Simulation Benefits"] = benefits

        try:
            paper["Additional link"] = result.find('div', {'class': "gs_or_ggsm"}).find('a')['href']
        except:
            paper["Additional link"] = ''


        """
        try:
            paper['Abstract'] = result.find('div', {'class': "gs_rs"}).text
        except:
            paper['Abstract'] = ''
        """

        for a in result.findAll('div', {'class': "gs_fl"})[-1].findAll('a'):
            if a.text != '':
                if a.text.startswith('Cited'):
                    paper['Cited by'] = a.text.rstrip().split()[-1]
                    paper['Cited list'] = url + a['href']
                if a.text.startswith('Related'):
                    paper['Related list'] = url + a['href']
                if a.text.startswith('Import'):
                    paper['Bibtex'] = requests.get(a['href']).text
                    
        papers.append(paper)
        # Wait 20 seconds until the next request to google
        time.sleep(5)

    return papers, len(html)






def generate_answer(question, insights):
    # This function will use ChatGPT to generate an answer based on the question and insights
    prompt = f"{question}\n\nTo answers use this information and nothing else: {', '.join(insights)}\n\n"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "user",
             "content": prompt}
        ]
    )

    return response.choices[0].text.strip()




def generate_report(types, applications, benefits):
    print("\nReport on LLMs in Social Simulations:")
    print("-" * 40)

    questions = [
        "What are the main types of applications of Social Simulations created using LLMs?",
        "What are the primary applications of LLMs in Social Simulations?",
        "What distinct benefits do LLMs offer over traditional methods in Social Simulations?",

    ]

    insights_lists = [types, applications, benefits]

    for i, question in enumerate(questions):
        answer = generate_answer(question, insights_lists[i])
        print(f"\n{i + 1}. {question}\n{answer}")








if __name__ == '__main__':
    query = config['search']['query']
    year = int(config['search']['start_year'])
    output = config['default']['result_path']

    logging.info("Starting...")
    logging.info("Result path: {}".format(output))
    logging.info("PICOC terms are: {}".format(picoc))
    logging.info("Search query is: {}".format(query))

    with open(output, 'a', newline='') as outcsv:
        csv.writer(outcsv).writerow(['Link', 'Additional link', 'Title', 'Authors', 'Abstract', 'Cited by',
                                     'Cited list', 'Related list', 'Bibtex', 'Year', 'Google page',"Is LLM Simulation", "Simulation Type", "Simulation Benefits"])

    # String search year by year.
    while year <= int(config['search']['end_year']):
        driver.get(url + "scholar?hl=en&q={0}&as_sdt=1&as_vis=1&as_ylo={1}&as_yhi={1}".format(query, year))
        check_captcha()
        page = 1
        total = 0
        while True:
            print("PARSER IN")
            art, t = parser(BeautifulSoup(driver.page_source, 'lxml'), page, year)
            print("PARSER OUT")
            total += t
            df = pd.DataFrame(art)
            df.to_csv(output, mode='a', header=False, index=False)
            try:
                driver.find_element_by_class_name("gs_ico_nav_next").click()
                check_captcha()
                page += 1
            except:
                logging.info(
                    "No more pages for {} year, total of {} pages and {} articles processed".format(year, page, total))
                year += 1
                # Wait 10 seconds until the next page request
                time.sleep(3)
                break

    types = list(df['Simulation Type'].dropna())
    applications = list(df['Application Type'].dropna())
    benefits = list(df['Simulation Benefits'].dropna())


    generate_report(types, applications, benefits)

    logging.info("Ending...")
    driver.close()
