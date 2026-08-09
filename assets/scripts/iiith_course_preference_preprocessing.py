"""
Script Description:
This script performs preprocessing on PDF files and saves the converted dataset to iiith_course_preferences.csv and iiith_course_mapping.csv

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

raw_data_dir = "../../data/temp"    # This will not be released since it contains PIIs

# This is to extract subject and all roll numbers using regex
subject_pattern = re.compile(r"Subject\s*:\s*(.+)")
roll_pattern = re.compile(r"\b\d{10}\b")

def normalize_subject(subject):
    """
    This removes the course code like MA2.101a
    This is important because in IIIT, some courses get fragmented into subcourses to accommodate the students
    One example is SMAI, which get two course codes: CS7.403a and CS7.403b
    Both fragments have the same syllabus, but are taken in different classes for convenience
    """
    subject = re.sub(r'^[A-Za-z]{2,}\d+\.\d+[a-zA-Z]?-', '', subject)
    subject = re.sub(r'\s+', ' ', subject).strip()

    return subject

def process_pdfs(base_path, student_to_cids, counter):
    """
    For a directory of PDFs, we obtain course-to-id and students-to-courses mapping
    """
    pdfs = os.listdir(base_path)
    course_to_cid = {}

    for pdf_name in pdfs:
        pdf_path = os.path.join(base_path, pdf_name)

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()

                if not text:
                    continue

                subject_match = subject_pattern.search(text)
                if not subject_match:
                    continue

                subject = subject_match.group(1).strip()    # For example: Subject: CS7.301-Machine, Data and Learning
                subject = normalize_subject(subject)    # For example: Subject: Machine, Data and Learning

                if (subject in course_to_cid.keys()):
                    course_id = course_to_cid[subject]
                else:
                    course_to_cid[subject] = counter
                    course_id = counter
                    counter += 1

                roll_numbers = roll_pattern.findall(text)
                for roll in roll_numbers:

                    if (roll in student_to_cids.keys()):
                        student_to_cids[roll].append(course_id)
                    else:
                        student_to_cids[roll] = [course_id]
    
    return course_to_cid, student_to_cids, counter

if __name__ == "__main__":
    output_students_csv_path = "../../data/iiith_course_preferences.csv"
    output_courses_csv_path = "../../data/iiith_course_mapping.csv"
    monsoon_path = os.path.join(raw_data_dir, "monsoon")
    spring_path = os.path.join(raw_data_dir, "spring")

    counter = 0
    student_to_cids = {}
    monsoon_course_to_cid, student_to_cids, counter = process_pdfs(monsoon_path, student_to_cids, counter)
    spring_course_to_cid, student_to_cids, counter = process_pdfs(spring_path, student_to_cids, counter)
    print(f"Total number of subjects: {counter}")
    print(f"Total number of students: {len(student_to_cids.keys())}")

    # Now we will create a dataframe with student roll number
    # This will make the data transactional i.e. similar to customer and the items bought by them
    # We will not include student roll in dataframe
    rows = []
    counter = 1
    for student in student_to_cids.keys():
        course_list = student_to_cids.get(student, [])
        courses_string = ",".join(map(str, course_list))

        rows.append({
            "student_id": counter,
            "courses": courses_string,
        })
        counter += 1

    df = pd.DataFrame(rows)
    df.to_csv(output_students_csv_path, index=False)

    rows = []
    for course in monsoon_course_to_cid.keys():
        rows.append({
            "cid": monsoon_course_to_cid[course],
            "course_name": course,
            "Semester": "Monsoon"
        })
    for course in spring_course_to_cid.keys():
            rows.append({
                "cid": spring_course_to_cid[course],
                "course_name": course,
                "Semester": "Spring"
            })

    df = pd.DataFrame(rows)
    df.set_index("cid", drop=False, inplace=True)    # We don't drop cid just for referencing later
    df.to_csv(output_courses_csv_path, index=False)