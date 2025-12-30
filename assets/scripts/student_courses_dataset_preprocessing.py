"""
Script Description:
This script performs preprocessing on raw student data and saves the converted dataset to data/student_courses.csv and data/course_mapping.json.

Preprocessing steps:
1. For each PDF, extract the subject from each page and map it to a unique course ID.
2. Extract all roll numbers on the page and assign them to the corresponding course.
3. Aggregate courses per student across all PDFs.
4. Save masked student course mappings to a CSV file and the course-to-ID mapping to a JSON file.
"""

import pdfplumber
import os
import re
import pandas as pd
import json
import hashlib
import random
from collections import defaultdict

username = "prit.kanadiya"
seed = int(hashlib.sha256(username.encode()).hexdigest(), 16) % (2**32)
rng = random.Random(seed)

raw_data_dir = "./raw_data"
output_csv = "./student_courses.csv"
course_mapping_path = "./course_mapping.json"

course_to_id = {}
id_to_course = {}
course_counter = 1
student_courses = defaultdict(set)

# This is to extract subject and all roll numbers that enrolled for that subject
subject_pattern = re.compile(r"Subject\s*:\s*(.+)")
roll_pattern = re.compile(r"\b\d{10}\b")

def get_course_id(course_name):
    global course_counter
    if course_name not in course_to_id:
        course_to_id[course_name] = course_counter
        id_to_course[course_counter] = course_name
        course_counter += 1
    return course_to_id[course_name]


for pdf_file in os.listdir(raw_data_dir):
    if not pdf_file.endswith(".pdf"):
        continue

    pdf_path = os.path.join(raw_data_dir, pdf_file)

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue

            subject_match = subject_pattern.search(text)
            if not subject_match:
                continue

            subject = subject_match.group(1).strip()
            course_id = get_course_id(subject)

            roll_numbers = roll_pattern.findall(text)

            for roll in roll_numbers:
                student_courses[roll].add(course_id)


student_ids = list(student_courses.keys())
rng.shuffle(student_ids)
masked_student_map = {}
masked_rows = []

for masked_id, original_roll in enumerate(student_ids):
    masked_student_map[masked_id] = original_roll
    course_list = ",".join(map(str, sorted(student_courses[original_roll])))
    masked_rows.append([masked_id, course_list])

df = pd.DataFrame(masked_rows, columns=["student_id", "course_list"])
df.to_csv(output_csv, index=False)

with open(course_mapping_path, "w", encoding="utf-8") as f:
    json.dump(id_to_course, f, indent=2)