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
        return extract_text(file_path, page_numbers=[0, 1,2,3])
    except requests.RequestException as e:
        print(f"Error downloading PDF: {e}")
        return ""
    except Exception as e:
        print(f"Error occurred while processing the PDF: {e}")
        return ""


import time


def analyze_abstract(abstract, max_retries=4, initial_delay=5):
    def request_to_openai(message_content, max_retries, initial_delay):
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": message_content}]
                )
                return response.choices[0].message['content'].strip()
            except Exception as e:
                if "The server is overloaded or not ready yet" in str(e) and retry_count < max_retries - 1:
                    wait_time = initial_delay * (2 ** retry_count)  # Exponential backoff
                    print(f"Server error detected. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    retry_count += 1
                else:
                    print("SMTH wrong in request_to_openai functions")
                    return "NA"
                    # raise  # Raise the exception if it's a different error or if retries are exhausted

    # Step 1: Check if the abstract discusses a simulation with LLM models
    response_1_content = request_to_openai(
        "Based on the paper abstract tell, whether main purpose of the study is to simulate or model social phenomena?"
        # "Social simulation is defined as a method that uses computer-based models to replicate, analyze,"
      #   "and predict complex social dynamics based on the behaviors and interactions of individuals and groups within a society"
        "Please give one word as an Yes/No/Unclear answer. Abstract: " + abstract,
        max_retries, initial_delay)
    print(f"Based on abstract: {response_1_content.lower()}")
    is_llm_simulation = response_1_content.lower() in ['yes', 'yes.',"unclear","unclear."]

    #  .
    if not is_llm_simulation:
        return "no", None, None, None, None


    
    # Step 15: Identify the domain of the simulation
    domain = request_to_openai(
        "What is the domain of the simulation described in this abstract (e.g., social, economic, etc.)? Please give max 5 words as an answer."
        "If there is no information to answer this question, state NA. Abstract: " + abstract, max_retries,
        initial_delay)

    # Step 2: Identify the type of simulation
    entities = request_to_openai(
        "Does the abstract mention entities that could represent individuals, groups, or institutions within a social context?"
        "Please give Yes/No/Unclear as an answer."
        "If there is no information to answer this question, state NA. Abstract: " + abstract, max_retries,
        initial_delay)

    # Step 3: Outlining the main benefits
    interactions = request_to_openai("Does the abstract mention interactions, behaviors, or dynamics that are characteristic of social systems ?"
                                  "Please give Yes/No/Unclear as an answer."
                                 "If there is no information to answer this question, state NA. Abstract: " + abstract,
                                 max_retries, initial_delay)
    """
    # Step 4: Outlining the main application
    applications = request_to_openai("What is the primary application of LLM in this social simulation?"
                                     "Describe shortly."
                                     "If there is no information to answer this question, state NA. Abstract: " + abstract,
                                     max_retries, initial_delay)
    """


    applications = None

    return response_1_content.lower(), domain, entities, applications, interactions


def analyze_title(title, max_retries=4, initial_delay=5):
    def request_to_openai(message_content, max_retries, initial_delay):
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": message_content}]
                )
                return response.choices[0].message['content'].strip()
            except Exception as e:
                if "The server is overloaded or not ready yet" in str(e) and retry_count < max_retries - 1:
                    wait_time = initial_delay * (2 ** retry_count)  # Exponential backoff
                    print(f"Server error detected. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    retry_count += 1
                else:
                    print("SMTH wrong in request_to_openai functions")
                    return "NA"
                    # raise  # Raise the exception if it's a different error or if retries are exhausted

    # Step 1: Check if the abstract discusses a simulation with LLM models
    response_1_content = request_to_openai(
        "Based on the scientific paper title tell, whether paper may describe a created by the authors computational model(s) or simulation that attempt to mimic social processes, behaviors, or systems?"
       # "Social simulation is defined as a method that uses computer-based models to replicate, analyze,"
     #   "and predict complex social dynamics based on the behaviors and interactions of individuals and groups within a society"
        "Please give one word as an Yes/No/Unclear answer. Title: " + title,
        max_retries, initial_delay)
    print(f"Based on title: {response_1_content.lower()}" )
    is_llm_simulation = response_1_content.lower() in ['yes', 'yes.',"unclear","unclear."]

    #  computational models that attempt to mimic social processes, behaviors, or systems.
    if not is_llm_simulation:
        return "no", None, None, None, None

    domain = None
    types = None

    benefits = None
    applications = None

    return response_1_content.lower(), domain, types, applications, benefits


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


# Sometimes a Captcha shows up. It needs to be fixed manually. This function makes the code wait until thisj be fixed
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
def parser(soup, page, year, number_simulations):
    papers = []
    html = soup.findAll('div', {'class': 'gs_r gs_or gs_scl'})
    for result in html:
        paper = {'Link': result.find('h3', {'class': "gs_rt"}).find('a')['href'], 'Additional link': '', 'Title': '',
                 'Authors': '', 'Abstract': '', 'Cited by': '', 'Cited list': '', 'Related list': '', 'Bibtex': '',
                 'Year': year, 'Google page': page, "Is LLM Simulation Title":'', "Is LLM Simulation Abstract":'', "Simulation Domain":'', "Entities?": '',"Application Type":'', "Interactions?":''}

        # If it does not pass at Title-Abstract-Keyword filter exclude this paper and continue
        """
        if not filterTitleAbsKey(paper['Link']):
            continue

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


        if (paper['Cited by'].isdigit() == False):
            continue




        paper['Title'] = result.find('h3', {'class': "gs_rt"}).text
        print(paper['Title'])
        paper['Authors'] = ";".join(
            ["%s:%s" % (a.text, a['href']) for a in result.find('div', {'class': "gs_a"}).findAll('a')])

        # Main logic

        """

        """
        is_llm_simulation_title, domain, entities, applications, interactions= analyze_title(paper["Title"])
        paper["Is LLM Simulation Title"] = is_llm_simulation_title
        if is_llm_simulation_title.lower() in ['yes', 'yes.',"unclear","unclear."]:
            #print(f"Based on title, it could be social simulation")
            if is_pdf(paper["Link"]):
                pdf_text = extract_text_from_pdf(paper["Link"])
                paper["Abstract"] = extract_abstract(pdf_text, content_type="pdf")
            else:
                driver.get(paper["Link"])
                time.sleep(1)
                papier_page_soup = BeautifulSoup(driver.page_source, 'html.parser')
                html_content = str(papier_page_soup)
                paper["Abstract"] = extract_abstract(html_content, content_type="html")
                driver.back()
                time.sleep(1)

            # Analyzing the abstract
            is_llm_simulation_abstract, domain, entities, applications, interactions = analyze_abstract(paper["Abstract"])

            if is_llm_simulation_abstract.lower() in ['yes', 'yes.',"unclear","unclear."]:
                #print(f"Based on abstract, it could be social simulation")
                number_simulations += 1
                print(f"We have collected {number_simulations} simulations")
           #else:
                #print(f"Based on abstract, it is not a social simulation")
            paper["Is LLM Simulation Abstract"] = is_llm_simulation_abstract


        # Storing the results in the paper dictionary

        print(f"Domain? {domain}")
        print(f"Entities? {entities}")
        print(f"Interactions? {interactions}")

        paper["Simulation Domain"] = domain
        paper["Entities?"] = entities
        paper["Application Type"] = applications
        paper["Interactions?"] = interactions

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



        papers.append(paper)


        # Wait 20 seconds until the next request to google
        time.sleep(2)

    return papers, len(html),number_simulations


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

    return response.choices[0].message['content'].strip()













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
                                     'Cited list', 'Related list', 'Bibtex', 'Year', 'Google page',"Is LLM Simulation Title","Is LLM Simulation Abstract","Simulation Domain", "Entities?","Application Type" ,"Interactions?"])

    # String search year by year.
    while year <= int(config['search']['end_year']):
        driver.get(url + "scholar?hl=en&q={0}&as_sdt=1&as_vis=1&as_ylo={1}&as_yhi={1}".format(query, year))
        check_captcha()
        page = 1
        total = 0
        number_simulations = 0
        while True:
            print("PARSER IN")

            art, t,number_simulations = parser(BeautifulSoup(driver.page_source, 'lxml'), page, year, number_simulations)

            print("PARSER OUT")
            total += t

            df = pd.DataFrame(art)
            df.to_csv(output, mode='a', header=False, index=False)

            try:
                driver.find_element_by_class_name("gs_ico_nav_next").click()
                print("ELEMENT WAS FOUND")
                check_captcha()
                print("CAPTCHA CHECKED")
                page += 1
                print("PAGE UPDATED")
            except:
                logging.info(
                    "No more pages for {} year, total of {} pages and {} articles processed".format(year, page, total))
                year += 1

                print("NO PAGE")
                # Wait 10 seconds until the next page request
                time.sleep(10)
                break


    logging.info("Ending...")
    driver.close()
