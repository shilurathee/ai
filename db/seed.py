import sqlite3
import random
import os
import datetime

# Seed for reproducibility
random.seed(42)

# Database path — created in the working directory
DB_PATH = 'dropout_predictor.db'


def main():
    # Remove existing database for idempotency (drop-and-recreate approach)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    # Connect to SQLite database (creates the file if it doesn't exist)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Read and execute schema.sql to create all 4 tables
    schema_path = os.path.join('db', 'schema.sql')
    with open(schema_path, 'r') as f:
        schema_sql = f.read()
    conn.executescript(schema_sql)

    # ── STEP B: Seed courses (fixed list) ──────────────────────────
    courses = [
        ('Mathematics', 4),
        ('Physics', 3),
        ('Programming', 4),
        ('English', 2),
        ('Data Structures', 4),
        ('Statistics', 3),
    ]
    cursor.executemany(
        "INSERT INTO courses (course_name, credits) VALUES (?, ?)",
        courses
    )

    # ── STEP C: Seed 25 students with realistic Indian names ──────
    names = [
        'Aarav Sharma', 'Priya Patel', 'Rohan Mehta', 'Sneha Iyer',
        'Karan Gupta', 'Ananya Nair', 'Vikram Singh', 'Deepa Reddy',
        'Arjun Kumar', 'Pooja Joshi', 'Rahul Das', 'Meera Pillai',
        'Siddharth Rao', 'Kavya Menon', 'Nikhil Verma', 'Divya Krishnan',
        'Aditya Bose', 'Shreya Ghosh', 'Manish Tiwari',
        'Lakshmi Subramaniam', 'Farhan Sheikh', 'Ritika Choudhary',
        'Aman Malhotra', 'Tanvi Saxena', 'Yash Pandey',
    ]

    # Date range for enrolled_date: 2022-06-01 to 2023-06-30
    start_date = datetime.date(2022, 6, 1)
    end_date = datetime.date(2023, 6, 30)
    date_range_days = (end_date - start_date).days

    students_data = []
    for name in names:
        # Generate email: firstname.lastname@college.edu (lowercase)
        parts = name.lower().split()
        email = parts[0] + '.' + parts[1] + '@college.edu'

        # Random enrolled_date within the range
        random_days = random.randint(0, date_range_days)
        enrolled = start_date + datetime.timedelta(days=random_days)
        enrolled_str = enrolled.strftime('%Y-%m-%d')

        students_data.append((name, email, enrolled_str))

    cursor.executemany(
        "INSERT INTO students (name, email, enrolled_date) VALUES (?, ?, ?)",
        students_data
    )

    # ── STEP D: Enroll each student in 3–5 random courses ─────────
    course_ids = [row[0] for row in cursor.execute("SELECT id FROM courses").fetchall()]
    student_ids = [row[0] for row in cursor.execute("SELECT id FROM students").fetchall()]
    semesters = ['Sem1', 'Sem2', 'Sem3']

    enrollment_records = []
    for sid in student_ids:
        # Pick 3 to 5 random courses for this student
        num_courses = random.randint(3, 5)
        chosen_courses = random.sample(course_ids, num_courses)
        for cid in chosen_courses:
            semester = random.choice(semesters)
            enrollment_records.append((sid, cid, semester))

    cursor.executemany(
        "INSERT INTO enrollments (student_id, course_id, semester) VALUES (?, ?, ?)",
        enrollment_records
    )

    # ── STEP E: Create assessments for each enrollment ────────────
    # Define assessment templates: (name, max_marks)
    assessment_templates = [
        ('Assignment 1', 20),
        ('Mid-term', 50),
        ('Assignment 2', 20),
        ('Final Exam', 100),
    ]

    # Categorize students into performance tiers
    num_students = len(student_ids)
    shuffled_ids = student_ids[:]
    random.shuffle(shuffled_ids)

    # 60% normal/good, 20% failing, 10% disengaged, 10% ghost
    normal_count = int(num_students * 0.6)   # 15 students
    failing_count = int(num_students * 0.2)  # 5 students
    disengaged_count = int(num_students * 0.1)  # 2-3 students
    ghost_count = num_students - normal_count - failing_count - disengaged_count

    normal_ids = set(shuffled_ids[:normal_count])
    failing_ids = set(shuffled_ids[normal_count:normal_count + failing_count])
    disengaged_ids = set(shuffled_ids[normal_count + failing_count:normal_count + failing_count + disengaged_count])
    ghost_ids = set(shuffled_ids[normal_count + failing_count + disengaged_count:])

    # Fetch all enrollments
    enrollments = cursor.execute("SELECT id, student_id FROM enrollments").fetchall()

    assessment_data = []
    for enrollment_id, student_id in enrollments:
        for assess_name, max_marks in assessment_templates:

            if student_id in normal_ids:
                # Normal/good students: marks between 50%–95% of max
                obtained = round(random.uniform(max_marks * 0.5, max_marks * 0.95), 1)

            elif student_id in failing_ids:
                # Failing students: marks below 40%, with 1–2 NULLs
                if random.random() < 0.35:
                    # ~35% chance of NULL (missing submission) — gives ~1-2 NULLs out of 4
                    obtained = None
                else:
                    obtained = round(random.uniform(0, max_marks * 0.4), 1)

            elif student_id in disengaged_ids:
                # Fully disengaged: 0 for all assessments
                obtained = 0.0

            else:
                # Ghost students: NULL for 3+ assessments, very low for the rest
                if random.random() < 0.75:
                    # ~75% chance of NULL — gives ~3 NULLs out of 4
                    obtained = None
                else:
                    obtained = round(random.uniform(0, max_marks * 0.3), 1)

            assessment_data.append((enrollment_id, assess_name, max_marks, obtained))

    cursor.executemany(
        "INSERT INTO assessments (enrollment_id, assessment_name, max_marks, obtained_marks) VALUES (?, ?, ?, ?)",
        assessment_data
    )

    # ── STEP F: Commit and confirm ────────────────────────────────
    conn.commit()

    # Print summary counts for verification
    student_count = cursor.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    course_count = cursor.execute("SELECT COUNT(*) FROM courses").fetchone()[0]
    enrollment_count = cursor.execute("SELECT COUNT(*) FROM enrollments").fetchone()[0]
    assessment_count = cursor.execute("SELECT COUNT(*) FROM assessments").fetchone()[0]

    conn.close()

    print("✅ Database created: " + DB_PATH)
    print("✅ " + str(course_count) + " courses inserted")
    print("✅ " + str(student_count) + " students inserted")
    print("✅ " + str(enrollment_count) + " enrollments created")
    print("✅ " + str(assessment_count) + " assessments seeded")
    print("✅ Ready for Phase 2")


if __name__ == '__main__':
    main()
