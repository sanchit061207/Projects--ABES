
import os
import sys
import pandas as pd
import numpy as np


# ============================================================
# SAVE OUTPUT IN TERMINAL AND TEXT FILE
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(
    BASE_DIR, "Data", "09_hospital_appointments.csv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "Output")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "Analysis_Results.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)


class Tee:
    """Display output in terminal and save it in a file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


original_stdout = sys.stdout
result_file = open(OUTPUT_FILE, "w", encoding="utf-8")
sys.stdout = Tee(original_stdout, result_file)


try:

    print("=" * 70)
    print("HOSPITAL APPOINTMENT OPERATIONS ANALYSIS")
    print("=" * 70)

    # ========================================================
    # TASK 1: LOAD AND INSPECT DATASET
    # ========================================================

    print("\nTASK 1: LOAD AND INSPECT DATASET")
    print("-" * 70)

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            "Dataset file not found! Please check Data/09_hospital_appointments.csv"
        )

    df = pd.read_csv(DATA_FILE)

    print("\nFirst 5 Records:")
    print(df.head())

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nDataset Information:")
    df.info()

    print("\nStatistical Description:")
    print(df.describe(include="all"))

    # ========================================================
    # TASK 2: HANDLE MISSING WAITING TIME VALUES
    # ========================================================

    print("\n\nTASK 2: HANDLE MISSING WAITING TIME VALUES")
    print("-" * 70)

    missing_values = df["Waiting_Time_Minutes"].isnull().sum()

    print("Missing Waiting Time Values:", missing_values)

    median_waiting_time = df["Waiting_Time_Minutes"].median()

    print("Median Waiting Time:", median_waiting_time)

    df["Waiting_Time_Minutes"] = (
        df["Waiting_Time_Minutes"].fillna(median_waiting_time)
    )

    print(
        "Missing Values After Handling:",
        df["Waiting_Time_Minutes"].isnull().sum()
    )

    # ========================================================
    # TASK 3: CHECK AND REMOVE DUPLICATE APPOINTMENT IDs
    # ========================================================

    print("\n\nTASK 3: CHECK DUPLICATE APPOINTMENT IDs")
    print("-" * 70)

    duplicate_count = df["Appointment_ID"].duplicated().sum()

    print("Number of Duplicate Appointment IDs:", duplicate_count)

    if duplicate_count > 0:
        print("\nDuplicate Records:")
        print(df[df["Appointment_ID"].duplicated(keep=False)])

        df = df.drop_duplicates(
            subset="Appointment_ID", keep="first"
        )

        print("\nDuplicate Records Removed.")

    print("Records After Removing Duplicates:", len(df))

    # ========================================================
    # TASK 4: TOTAL APPOINTMENTS AND ATTENDANCE PERCENTAGES
    # ========================================================

    print("\n\nTASK 4: APPOINTMENT AND ATTENDANCE ANALYSIS")
    print("-" * 70)

    total_appointments = len(df)

    print("Total Appointments:", total_appointments)

    attendance_counts = df["Attendance_Status"].value_counts()

    attendance_percentages = (
        df["Attendance_Status"].value_counts(normalize=True) * 100
    )

    attendance_summary = pd.DataFrame({
        "Count": attendance_counts,
        "Percentage": attendance_percentages.round(2)
    })

    print("\nAttendance Summary:")
    print(attendance_summary)

    # ========================================================
    # TASK 5: AVERAGE WAITING TIME BY DEPARTMENT
    # ========================================================

    print("\n\nTASK 5: AVERAGE WAITING TIME BY DEPARTMENT")
    print("-" * 70)

    department_waiting = (
        df.groupby("Department")["Waiting_Time_Minutes"]
        .mean()
        .round(2)
        .sort_values(ascending=False)
    )

    print("\nAverage Waiting Time:")
    print(department_waiting)

    # ========================================================
    # TASK 6: DEPARTMENTS WITH HIGHEST NO-SHOW PERCENTAGE
    # ========================================================

    print("\n\nTASK 6: DEPARTMENT-WISE NO-SHOW ANALYSIS")
    print("-" * 70)

    department_no_show = (
        df.assign(
            No_Show=df["Attendance_Status"].eq("No-show")
        )
        .groupby("Department")["No_Show"]
        .mean()
        .mul(100)
        .round(2)
        .sort_values(ascending=False)
    )

    print("\nNo-Show Percentage by Department:")
    print(department_no_show)

    if not department_no_show.empty:
        print(
            "\nDepartment with Highest No-Show Percentage:",
            department_no_show.idxmax()
        )

        print(
            "Highest No-Show Percentage:",
            department_no_show.max(), "%"
        )

    # ========================================================
    # TASK 7: SMS REMINDER AND ATTENDANCE COMPARISON
    # ========================================================

    print("\n\nTASK 7: SMS REMINDER ANALYSIS")
    print("-" * 70)

    sms_analysis = pd.crosstab(
        df["SMS_Reminder"],
        df["Attendance_Status"]
    )

    print("\nSMS Reminder vs Attendance:")
    print(sms_analysis)

    sms_percentage = pd.crosstab(
        df["SMS_Reminder"],
        df["Attendance_Status"],
        normalize="index"
    ) * 100

    print("\nAttendance Percentage by SMS Reminder:")
    print(sms_percentage.round(2))

    # ========================================================
    # TASK 8: AVERAGE CONSULTATION FEE BY DEPARTMENT
    # ========================================================

    print("\n\nTASK 8: CONSULTATION FEE BY DEPARTMENT")
    print("-" * 70)

    fee_analysis = df.groupby("Department")[
        "Consultation_Fee"
    ].agg(["mean", "sum", "count"])

    fee_analysis.columns = [
        "Average_Fee",
        "Total_Fee",
        "Appointment_Count"
    ]

    fee_analysis = fee_analysis.round(2)

    print("\nDepartment-wise Consultation Fee:")
    print(fee_analysis)

    # ========================================================
    # TASK 9: APPOINTMENT VOLUME BY DATE AND DEPARTMENT
    # ========================================================

    print("\n\nTASK 9: APPOINTMENT VOLUME ANALYSIS")
    print("-" * 70)

    df["Appointment_Date"] = pd.to_datetime(
        df["Appointment_Date"], errors="coerce"
    )

    appointment_volume = pd.crosstab(
        df["Appointment_Date"].dt.strftime("%Y-%m-%d"),
        df["Department"]
    )

    print("\nAppointment Count by Date and Department:")
    print(appointment_volume)

    # ========================================================
    # TASK 10: NEW VS FOLLOW-UP APPOINTMENT WAITING TIMES
    # ========================================================

    print("\n\nTASK 10: APPOINTMENT TYPE WAITING TIME ANALYSIS")
    print("-" * 70)

    appointment_type_analysis = (
        df.groupby("Appointment_Type")["Waiting_Time_Minutes"]
        .agg(["mean", "median", "min", "max"])
        .round(2)
    )

    print("\nWaiting Time Comparison:")
    print(appointment_type_analysis)

    # ========================================================
    # TASK 11: NUMPY STATISTICAL ANALYSIS
    # ========================================================

    print("\n\nTASK 11: NUMPY STATISTICAL ANALYSIS")
    print("-" * 70)

    waiting_times = df["Waiting_Time_Minutes"].dropna().to_numpy()

    median = np.median(waiting_times)
    q1 = np.percentile(waiting_times, 25)
    q2 = np.percentile(waiting_times, 50)
    q3 = np.percentile(waiting_times, 75)
    standard_deviation = np.std(waiting_times)

    print("Median Waiting Time:", median)
    print("First Quartile (Q1):", q1)
    print("Second Quartile (Q2):", q2)
    print("Third Quartile (Q3):", q3)
    print("Standard Deviation:", round(standard_deviation, 2))

    # ========================================================
    # TASK 12: IDENTIFY UNUSUALLY LONG WAITING TIMES
    # ========================================================

    print("\n\nTASK 12: LONG WAITING TIME OUTLIER ANALYSIS")
    print("-" * 70)

    iqr = q3 - q1

    lower_limit = q1 - (1.5 * iqr)
    upper_limit = q3 + (1.5 * iqr)

    print("Interquartile Range:", iqr)
    print("Lower Limit:", lower_limit)
    print("Upper Limit:", upper_limit)

    outliers = df[
        (df["Waiting_Time_Minutes"] < lower_limit) |
        (df["Waiting_Time_Minutes"] > upper_limit)
    ]

    print("\nUnusually Long Waiting Time Records:")
    long_waiting = df[
        df["Waiting_Time_Minutes"] > upper_limit
    ]

    if long_waiting.empty:
        print("No unusually long waiting times found.")
    else:
        print(
            long_waiting[
                ["Appointment_ID", "Department", "Waiting_Time_Minutes"]
            ]
        )

    print("\nTotal Outlier Records:", len(outliers))

    # ========================================================
    # TASK 13: WAITING TIME CATEGORIES
    # ========================================================

    print("\n\nTASK 13: WAITING TIME CATEGORY ANALYSIS")
    print("-" * 70)

    bins = [-np.inf, 15, 30, 60, np.inf]

    labels = [
        "Short (0-15)",
        "Moderate (16-30)",
        "Long (31-60)",
        "Very Long (>60)"
    ]

    df["Waiting_Time_Category"] = pd.cut(
        df["Waiting_Time_Minutes"],
        bins=bins,
        labels=labels
    )

    waiting_category_counts = (
        df["Waiting_Time_Category"].value_counts(sort=False)
    )

    print("\nWaiting Time Categories:")
    print(waiting_category_counts)

    # ========================================================
    # TASK 14: ESTIMATED FEES FOR ATTENDED APPOINTMENTS
    # ========================================================

    print("\n\nTASK 14: ATTENDED APPOINTMENT FEE ANALYSIS")
    print("-" * 70)

    attended_df = df[
        df["Attendance_Status"] == "Attended"
    ]

    attended_total_fee = attended_df[
        "Consultation_Fee"
    ].sum()

    attended_average_fee = attended_df[
        "Consultation_Fee"
    ].mean()

    attended_department_fee = (
        attended_df.groupby("Department")["Consultation_Fee"]
        .agg(["count", "sum", "mean"])
        .round(2)
    )

    print("Total Attended Appointments:", len(attended_df))

    print(
        "Estimated Total Consultation Fees:",
        round(attended_total_fee, 2)
    )

    print(
        "Average Consultation Fee:",
        round(attended_average_fee, 2)
    )

    print("\nDepartment-wise Attended Appointment Fees:")
    print(attended_department_fee)

    # ========================================================
    # TASK 15: OPERATIONAL RECOMMENDATIONS
    # ========================================================

    print("\n\nTASK 15: OPERATIONAL RECOMMENDATIONS")
    print("-" * 70)

    print("\n1. Appointment Scheduling")
    print(
        "   Review appointment schedules in departments "
        "with higher average waiting times."
    )

    print("\n2. No-Show Reduction")
    print(
        "   Pay attention to departments with higher no-show "
        "percentages and review their appointment processes."
    )

    print("\n3. SMS Reminders")
    print(
        "   Continue monitoring attendance for appointments "
        "with and without SMS reminders."
    )

    print("\n4. Waiting Time Management")
    print(
        "   Review appointments with unusually long waiting "
        "times to identify possible scheduling delays."
    )

    print("\n5. Fee Monitoring")
    print(
        "   Regularly review department-wise consultation "
        "fee records for better operational tracking."
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("PROJECT SUMMARY")
    print("=" * 70)

    print("Total Records Analysed:", len(df))
    print("Total Departments:", df["Department"].nunique())
    print("Total Attended Appointments:", len(attended_df))
    print("Total No-Show Appointments:",
          (df["Attendance_Status"] == "No-show").sum())
    print("Total Cancelled Appointments:",
          (df["Attendance_Status"] == "Cancelled").sum())

    print("\nAll 15 Tasks Completed Successfully!")

    print("=" * 70)

finally:

    # Restore normal terminal output
    sys.stdout = original_stdout

    # Close the output file
    result_file.close()


print("\nResults saved successfully!")
print("Output file:", OUTPUT_FILE)
