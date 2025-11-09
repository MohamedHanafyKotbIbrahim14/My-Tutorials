import streamlit as st
import pandas as pd
import numpy as np
import pulp
import re
from typing import Dict, List, Tuple, Set
import plotly.express as px
import plotly.graph_objects as go
from datetime import time as dt_time

class TutorAssignmentLP:
    """Tutor-Class Assignment using Linear Programming with time conflict detection."""
    
    def __init__(self, classes_df: pd.DataFrame, tutors_df: pd.DataFrame, 
                 preferences: Dict[Tuple[str, str], float],
                 tutor_max_classes: Dict[str, int]):
        """
        Initialize the LP problem.
        
        Args:
            classes_df: DataFrame with columns [course, class_id, section, time, mode]
            tutors_df: DataFrame with tutor information
            preferences: Dict mapping (tutor, course) -> preference_score
            tutor_max_classes: Dict mapping tutor -> max_classes
        """
        self.classes_df = classes_df
        self.tutors_df = tutors_df
        self.preferences = preferences
        self.tutor_max_classes = tutor_max_classes
        
        self.tutors = list(tutors_df['tutor_name'].unique())
        self.courses = list(classes_df['course'].unique())
        
        self.model = None
        self.x_vars = {}
        self.time_conflicts = self._detect_time_conflicts()
        
    def _detect_time_conflicts(self) -> Dict[Tuple[str, str], bool]:
        """
        Detect time conflicts between classes.
        Returns dict mapping (class_id1, class_id2) -> True if they conflict.
        """
        conflicts = {}
        
        # Parse time strings and detect overlaps
        classes_with_times = []
        for idx, row in self.classes_df.iterrows():
            time_str = str(row['time'])
            parsed_times = self._parse_time_string(time_str)
            if parsed_times:
                classes_with_times.append({
                    'class_id': row['class_id'],
                    'times': parsed_times
                })
        
        # Check all pairs for conflicts
        for i, class1 in enumerate(classes_with_times):
            for class2 in classes_with_times[i+1:]:
                if self._times_overlap(class1['times'], class2['times']):
                    conflicts[(class1['class_id'], class2['class_id'])] = True
                    conflicts[(class2['class_id'], class1['class_id'])] = True
        
        return conflicts
    
    def _parse_time_string(self, time_str: str) -> List[Dict]:
        """
        Parse time string like 'Fri 09-10:30(w1-5, 7-10, Col LG01)'
        Returns list of dicts with day, start_time, end_time
        """
        if pd.isna(time_str) or time_str == 'nan':
            return []
        
        time_slots = []
        
        # Pattern to match: Day HH:MM-HH:MM or Day HH-HH:MM
        pattern = r'(Mon|Tue|Wed|Thu|Fri)\s+(\d{1,2})(?::(\d{2}))?-(\d{1,2})(?::(\d{2}))?'
        
        for match in re.finditer(pattern, time_str):
            day = match.group(1)
            start_hour = int(match.group(2))
            start_min = int(match.group(3)) if match.group(3) else 0
            end_hour = int(match.group(4))
            end_min = int(match.group(5)) if match.group(5) else 0
            
            time_slots.append({
                'day': day,
                'start': start_hour * 60 + start_min,  # Convert to minutes
                'end': end_hour * 60 + end_min
            })
        
        return time_slots
    
    def _times_overlap(self, times1: List[Dict], times2: List[Dict]) -> bool:
        """Check if two time slot lists overlap."""
        for t1 in times1:
            for t2 in times2:
                # Same day and overlapping times
                if t1['day'] == t2['day']:
                    if not (t1['end'] <= t2['start'] or t2['end'] <= t1['start']):
                        return True
        return False
    
    def build_model(self):
        """Build the LP model."""
        self.model = pulp.LpProblem("Tutor_Class_Assignment", pulp.LpMaximize)
        
        # Decision variables: x[(tutor, course, class_id)] = 1 if assigned
        self.x_vars = {}
        for tutor in self.tutors:
            for idx, row in self.classes_df.iterrows():
                course = row['course']
                class_id = row['class_id']
                
                # Only create variable if tutor has preference for this course
                if self.preferences.get((tutor, course), 0) > 0:
                    self.x_vars[(tutor, course, class_id)] = pulp.LpVariable(
                        f"x_{tutor}_{course}_{class_id}",
                        cat='Binary'
                    )
        
        # Objective: Maximize total preference satisfaction
        objective = pulp.lpSum([
            self.preferences.get((tutor, course), 0) * self.x_vars[(tutor, course, class_id)]
            for (tutor, course, class_id) in self.x_vars.keys()
        ])
        self.model += objective
        
        self._add_constraints()
    
    def _add_constraints(self):
        """Add all constraints to the model."""
        
        # Constraint 1: Each class must have exactly ONE tutor
        for idx, row in self.classes_df.iterrows():
            course = row['course']
            class_id = row['class_id']
            
            assigned_tutors = pulp.lpSum([
                self.x_vars[(tutor, course, class_id)]
                for tutor in self.tutors
                if (tutor, course, class_id) in self.x_vars
            ])
            
            self.model += (
                assigned_tutors == 1,
                f"Class_Coverage_{course}_{class_id}"
            )
        
        # Constraint 2: Tutor workload limit
        for tutor in self.tutors:
            total_classes = pulp.lpSum([
                self.x_vars[(tutor, course, class_id)]
                for (t, course, class_id) in self.x_vars.keys()
                if t == tutor
            ])
            
            max_classes = self.tutor_max_classes.get(tutor, 3)
            self.model += (
                total_classes <= max_classes,
                f"Tutor_Workload_{tutor}"
            )
        
        # Constraint 3: Time conflict prevention
        # If tutor is assigned to two classes that conflict, sum must be ≤ 1
        for tutor in self.tutors:
            # Get all classes this tutor could be assigned to
            tutor_possible_classes = [
                (course, class_id) 
                for (t, course, class_id) in self.x_vars.keys() 
                if t == tutor
            ]
            
            # Check all pairs for conflicts
            for i, (course1, class1) in enumerate(tutor_possible_classes):
                for course2, class2 in tutor_possible_classes[i+1:]:
                    if (class1, class2) in self.time_conflicts:
                        # These two classes conflict - tutor can't teach both
                        self.model += (
                            self.x_vars[(tutor, course1, class1)] + 
                            self.x_vars[(tutor, course2, class2)] <= 1,
                            f"Time_Conflict_{tutor}_{class1}_{class2}"
                        )
    
    def solve(self):
        """Solve the optimization problem."""
        if self.model is None:
            self.build_model()
        
        solver = pulp.PULP_CBC_CMD(msg=0)
        self.model.solve(solver)
        
        status = pulp.LpStatus[self.model.status]
        
        if status == 'Optimal':
            return self._extract_solution()
        else:
            return {
                'status': status,
                'objective_value': None,
                'assignments': {},
                'tutor_loads': {},
                'unassigned_classes': []
            }
    
    def _extract_solution(self):
        """Extract solution from solved model."""
        assignments = {}  # (course, class_id) -> tutor
        tutor_loads = {tutor: {'classes': [], 'total': 0} for tutor in self.tutors}
        unassigned_classes = []
        
        # Extract assignments
        for (tutor, course, class_id), var in self.x_vars.items():
            if var.varValue == 1:
                assignments[(course, class_id)] = tutor
                tutor_loads[tutor]['classes'].append((course, class_id))
                tutor_loads[tutor]['total'] += 1
        
        # Find unassigned classes
        for idx, row in self.classes_df.iterrows():
            course = row['course']
            class_id = row['class_id']
            if (course, class_id) not in assignments:
                unassigned_classes.append((course, class_id))
        
        return {
            'status': 'Optimal',
            'objective_value': pulp.value(self.model.objective),
            'assignments': assignments,
            'tutor_loads': tutor_loads,
            'unassigned_classes': unassigned_classes
        }


def extract_course_codes(text: str) -> List[str]:
    """Extract course codes from preference text."""
    if pd.isna(text) or str(text).strip() == '':
        return []
    
    # Pattern to match course codes like ACTL2102, RISK5001, COMM2501/5501
    pattern = r'\b(ACTL|RISK|COMM)\d{4}(?:/\d{4})?\b'
    # Use finditer to get full matches, not just the captured group
    codes = [match.group() for match in re.finditer(pattern, str(text))]
    
    return list(set(codes))


def parse_max_classes(value) -> Tuple[int, str]:
    """
    Parse max_classes value from various formats.
    Returns: (parsed_value, status_message)
    """
    if pd.isna(value) or str(value).strip() == '':
        return 3, "Empty (defaulted to 3)"
    
    value_str = str(value).strip()
    
    # Handle datetime objects (dates entered by mistake)
    if isinstance(value, pd.Timestamp) or 'Timestamp' in str(type(value)):
        return 3, f"Date value '{value}' (defaulted to 3)"
    
    # Check if it looks like a date string (YYYY-MM-DD or similar with year > 1000)
    if re.search(r'\b(19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b', value_str):
        return 3, f"Date value '{value_str}' (defaulted to 3)"
    
    # Try direct integer conversion
    try:
        num = int(float(value_str))
        if num > 100:  # Likely a year or invalid number
            return 3, f"Invalid large number '{value_str}' (defaulted to 3)"
        return num, "OK"
    except:
        pass
    
    # Handle ranges like "2-3" (but not dates)
    if '-' in value_str and not re.search(r'\d{4}', value_str):
        try:
            parts = value_str.split('-')
            low = int(parts[0].strip())
            high = int(parts[1].strip())
            if low <= 10 and high <= 10:  # Reasonable range for classes
                avg = (low + high) // 2
                return avg, f"Range {value_str} (using average: {avg})"
        except:
            pass
    
    # Handle "3+" or "2+"
    if '+' in value_str:
        try:
            num = int(value_str.replace('+', '').strip())
            if num <= 10:
                return num, f"'{value_str}' (using {num})"
        except:
            pass
    
    # Extract first reasonable number from the string (< 100)
    numbers = re.findall(r'\b\d+\b', value_str)
    for num_str in numbers:
        num = int(num_str)
        if num <= 10:  # Reasonable class count
            return num, f"Extracted {num} from '{value_str}'"
    
    # Default fallback
    return 3, f"Could not parse '{value_str}' (defaulted to 3)"


def load_file1_classes(file_path: str) -> pd.DataFrame:
    """Load and parse File 1 (Classes for T3)."""
    df = pd.read_excel(file_path, header=None)
    
    classes_data = []
    current_course = None
    
    for idx, row in df.iterrows():
        val = str(row[0]).strip()
        
        # Check if this is a course header row
        if val.startswith('ACTL') or val.startswith('RISK') or val.startswith('COMM'):
            if pd.isna(row[1]) or 'Class' not in str(row[1]):
                current_course = val
                continue
        
        # Check if this is a class row (has class number and type)
        if current_course and pd.notna(row[0]) and pd.notna(row[1]):
            class_type = str(row[1]).strip()
            
            # Only include TUT and LAB (skip LEC)
            if class_type in ['TUT', 'LAB']:
                classes_data.append({
                    'course': current_course,
                    'class_id': str(row[0]),
                    'type': class_type,
                    'section': str(row[2]) if pd.notna(row[2]) else '',
                    'mode': str(row[3]) if pd.notna(row[3]) else '',
                    'status': str(row[4]) if pd.notna(row[4]) else '',
                    'time': str(row[5]) if pd.notna(row[5]) else '',
                    'tutor': str(row[6]) if pd.notna(row[6]) else ''
                })
    
    return pd.DataFrame(classes_data)


def load_file2_tutors(file_path: str) -> Tuple[pd.DataFrame, Dict[str, List[str]], Dict[str, int], Dict[str, str]]:
    """
    Load and parse File 2 (Tutor Preferences).
    Returns: (tutors_df, preferences_dict, max_classes_dict, parsing_status)
    """
    df = pd.read_excel(file_path, sheet_name=0)
    
    # Column indices
    first_name_col = df.columns[6]
    last_name_col = df.columns[8]
    pref_name_col = df.columns[10]
    t3_pref_col = df.columns[108]  # DE - T3 preferences
    max_classes_col = df.columns[110]  # DG - Max classes (tutorials per session)
    
    tutors_data = []
    preferences = {}
    max_classes = {}
    parsing_status = {}
    
    for idx, row in df.iterrows():
        # Get tutor name
        pref_name = row[pref_name_col]
        if pd.isna(pref_name) or str(pref_name).strip() == '':
            first = str(row[first_name_col]).strip() if pd.notna(row[first_name_col]) else ''
            last = str(row[last_name_col]).strip() if pd.notna(row[last_name_col]) else ''
            tutor_name = f"{first} {last}".strip()
        else:
            tutor_name = str(pref_name).strip()
        
        if not tutor_name or tutor_name == '':
            continue
        
        # Get T3 preferences
        t3_pref_text = row[t3_pref_col]
        t3_courses_raw = extract_course_codes(t3_pref_text)
        
        # Expand dual codes (e.g., "ACTL3162/5106" -> ["ACTL3162", "ACTL5106"])
        t3_courses_expanded = []
        for code in t3_courses_raw:
            if '/' in code:
                # Split dual code: "ACTL3162/5106" -> ["ACTL3162", "ACTL5106"]
                parts = code.split('/')
                base = parts[0]  # e.g., "ACTL3162"
                prefix = base[:4]  # e.g., "ACTL"
                second_code = prefix + parts[1]  # e.g., "ACTL5106"
                t3_courses_expanded.append(base)
                t3_courses_expanded.append(second_code)
            else:
                t3_courses_expanded.append(code)
        
        # Remove duplicates
        t3_courses_expanded = list(set(t3_courses_expanded))
        
        # Only include tutors who have T3 preferences
        if len(t3_courses_expanded) > 0:
            # Parse max classes
            max_val, status = parse_max_classes(row[max_classes_col])
            
            tutors_data.append({
                'tutor_name': tutor_name,
                'first_name': str(row[first_name_col]) if pd.notna(row[first_name_col]) else '',
                'last_name': str(row[last_name_col]) if pd.notna(row[last_name_col]) else '',
                'email': str(row[df.columns[9]]) if pd.notna(row[df.columns[9]]) else '',
                't3_preferences_raw': str(t3_pref_text),
                't3_courses': ', '.join(sorted(t3_courses_expanded)),
                'max_classes': max_val,
                'max_classes_status': status
            })
            
            preferences[tutor_name] = t3_courses_expanded
            max_classes[tutor_name] = max_val
            parsing_status[tutor_name] = status
    
    return pd.DataFrame(tutors_data), preferences, max_classes, parsing_status


def main():
    st.set_page_config(
        page_title="Tutor Assignment Optimizer",
        page_icon="👨‍🏫",
        layout="wide"
    )
    
    st.title("👨‍🏫 Tutor-Class Assignment Optimizer (T3)")
    st.markdown("**Linear Programming Solution with Time Conflict Detection**")
    st.markdown("---")
    
    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 1
    
    # Step 1: File Upload
    if st.session_state.step == 1:
        show_upload_step()
    
    # Step 2: Review & Correct Max Classes
    elif st.session_state.step == 2:
        show_review_step()
    
    # Step 3: Analysis & Visualization
    elif st.session_state.step == 3:
        show_analysis_step()
    
    # Step 4: Run Optimization & Results
    elif st.session_state.step == 4:
        show_optimization_step()


def show_upload_step():
    """Step 1: Upload both files."""
    st.header("Step 1: Upload Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 File 1: Classes (T3)")
        st.markdown("""
        **Required structure:**
        - Course codes in first column
        - Class details: Number, Type (TUT/LAB), Section, Mode, Status, Time, Tutor
        - Only TUT/LAB classes will be used (LEC ignored)
        """)
        
        file1 = st.file_uploader(
            "Upload File 1 (Classes)",
            type=['xlsx', 'xls'],
            key='file1'
        )
    
    with col2:
        st.subheader("👥 File 2: Tutor Preferences")
        st.markdown("""
        **Will extract:**
        - Column DE (108): T3 course preferences
        - Column DG (109): Maximum classes per week
        - Tutor names (First, Last, Preferred)
        """)
        
        file2 = st.file_uploader(
            "Upload File 2 (Tutor Preferences)",
            type=['xlsx', 'xls'],
            key='file2'
        )
    
    if file1 is not None and file2 is not None:
        with st.spinner("Processing files..."):
            try:
                # Load classes
                classes_df = load_file1_classes(file1)
                st.session_state.classes_df = classes_df
                
                # Load tutors
                tutors_df, preferences, max_classes, parsing_status = load_file2_tutors(file2)
                st.session_state.tutors_df = tutors_df
                st.session_state.preferences = preferences
                st.session_state.max_classes = max_classes
                st.session_state.parsing_status = parsing_status
                
                # Show preview
                st.success("✅ Files loaded successfully!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Classes Found (TUT/LAB)", len(classes_df))
                    st.metric("Unique Courses", classes_df['course'].nunique())
                
                with col2:
                    st.metric("Tutors with T3 Preferences", len(tutors_df))
                    
                # Show preview
                with st.expander("📋 Preview Classes Data"):
                    st.dataframe(classes_df.head(20), use_container_width=True)
                
                with st.expander("👥 Preview Tutors Data"):
                    st.dataframe(tutors_df[['tutor_name', 't3_courses', 'max_classes', 'max_classes_status']].head(20), 
                               use_container_width=True)
                
                if st.button("Next: Review Max Classes →", type="primary"):
                    st.session_state.step = 2
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Error processing files: {str(e)}")
                st.exception(e)


def show_review_step():
    """Step 2: Review and correct max_classes values."""
    st.header("Step 2: Review & Correct Max Classes")
    
    tutors_df = st.session_state.tutors_df
    parsing_status = st.session_state.parsing_status
    
    st.markdown("""
    **Review the parsed 'Maximum Classes' values below.**  
    - ✅ Green = Successfully parsed
    - ⚠️ Yellow = Parsed with assumptions (review recommended)
    - ❌ Red = Failed to parse (defaulted to 3)
    
    You can edit any value in the table below.
    """)
    
    # Create editable dataframe
    edit_df = tutors_df[['tutor_name', 'max_classes', 'max_classes_status']].copy()
    
    # Show which ones need attention
    needs_review = edit_df[edit_df['max_classes_status'].str.contains('defaulted|Could not parse', case=False, na=False)]
    
    if len(needs_review) > 0:
        st.warning(f"⚠️ {len(needs_review)} tutor(s) have values that may need correction (highlighted below)")
    else:
        st.success("✅ All max_classes values parsed successfully!")
    
    st.markdown("---")
    
    # Show editable table
    st.subheader("Edit Maximum Classes")
    
    edited_data = []
    
    for idx, row in edit_df.iterrows():
        col1, col2, col3, col4 = st.columns([3, 2, 3, 2])
        
        with col1:
            st.text(row['tutor_name'])
        
        with col2:
            status = row['max_classes_status']
            if 'defaulted' in status.lower() or 'could not parse' in status.lower():
                st.error("❌ Needs review")
            elif 'range' in status.lower() or 'extracted' in status.lower():
                st.warning("⚠️ Check value")
            else:
                st.success("✅ OK")
        
        with col3:
            st.text(status)
        
        with col4:
            new_value = st.number_input(
                "Max",
                min_value=1,
                max_value=10,
                value=int(row['max_classes']),
                key=f"max_{idx}",
                label_visibility="collapsed"
            )
            edited_data.append((row['tutor_name'], new_value))
    
    st.markdown("---")
    
    # Update session state with edited values
    new_max_classes = {tutor: val for tutor, val in edited_data}
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Back to Upload"):
            st.session_state.step = 1
            st.rerun()
    
    with col3:
        if st.button("Next: Analysis →", type="primary"):
            st.session_state.max_classes = new_max_classes
            st.session_state.step = 3
            st.rerun()


def show_analysis_step():
    """Step 3: Show data analysis and visualizations."""
    st.header("Step 3: Data Analysis")
    
    classes_df = st.session_state.classes_df
    tutors_df = st.session_state.tutors_df
    preferences = st.session_state.preferences
    max_classes = st.session_state.max_classes
    
    # Summary metrics
    st.subheader("📊 Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Classes", len(classes_df))
    with col2:
        st.metric("Unique Courses", classes_df['course'].nunique())
    with col3:
        st.metric("Available Tutors", len(tutors_df))
    with col4:
        total_capacity = sum(max_classes.values())
        st.metric("Total Tutor Capacity", total_capacity)
    
    # Check feasibility
    if total_capacity < len(classes_df):
        st.error(f"⚠️ WARNING: Total tutor capacity ({total_capacity}) is less than total classes ({len(classes_df)}). Some classes may remain unassigned!")
    else:
        surplus = total_capacity - len(classes_df)
        st.success(f"✅ Sufficient capacity: {surplus} extra class slots available")
    
    st.markdown("---")
    
    # Classes per course
    st.subheader("📚 Classes per Course")
    course_counts = classes_df.groupby('course').size().reset_index(name='count')
    course_counts = course_counts.sort_values('count', ascending=False)
    
    fig_courses = px.bar(
        course_counts,
        x='course',
        y='count',
        title='Number of Classes (TUT/LAB) per Course',
        labels={'course': 'Course', 'count': 'Number of Classes'},
        color='count',
        color_continuous_scale='Blues'
    )
    fig_courses.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_courses, use_container_width=True)
    
    # Build preference matrix
    st.subheader("🎯 Tutor-Course Preference Matrix")
    
    courses = sorted(classes_df['course'].unique())
    tutors = sorted(tutors_df['tutor_name'].unique())
    
    pref_matrix = pd.DataFrame(0, index=tutors, columns=courses)
    
    for tutor in tutors:
        tutor_courses = preferences.get(tutor, [])
        for course in tutor_courses:
            if course in pref_matrix.columns:
                pref_matrix.loc[tutor, course] = 10
    
    fig_pref = px.imshow(
        pref_matrix.values,
        labels=dict(x="Course", y="Tutor", color="Preference"),
        x=pref_matrix.columns.tolist(),
        y=pref_matrix.index.tolist(),
        color_continuous_scale="RdYlGn",
        title="Tutor Preferences for Courses (10 = Preferred, 0 = Not Preferred)",
        aspect="auto"
    )
    fig_pref.update_layout(
        height=max(600, len(tutors) * 15),
        xaxis=dict(tickangle=90),
        yaxis=dict(tickfont=dict(size=9))
    )
    st.plotly_chart(fig_pref, use_container_width=True)
    
    # Course coverage analysis
    st.subheader("📋 Course Coverage Analysis")
    
    coverage_data = []
    for course in courses:
        num_classes = len(classes_df[classes_df['course'] == course])
        num_tutors = (pref_matrix[course] > 0).sum()
        total_capacity_for_course = sum([
            max_classes.get(tutor, 0) 
            for tutor in tutors 
            if pref_matrix.loc[tutor, course] > 0
        ])
        
        coverage_data.append({
            'Course': course,
            'Classes': num_classes,
            'Qualified Tutors': num_tutors,
            'Tutor Capacity': total_capacity_for_course,
            'Ratio': f"{total_capacity_for_course / num_classes:.2f}x" if num_classes > 0 else "N/A",
            'Status': '✅' if total_capacity_for_course >= num_classes else '⚠️'
        })
    
    coverage_df = pd.DataFrame(coverage_data)
    st.dataframe(coverage_df, use_container_width=True, hide_index=True)
    
    # Highlight problematic courses
    problematic = coverage_df[coverage_df['Status'] == '⚠️']
    if len(problematic) > 0:
        st.warning(f"⚠️ {len(problematic)} course(s) have insufficient tutor capacity:")
        st.dataframe(problematic, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Navigation
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back to Review"):
            st.session_state.step = 2
            st.rerun()
    
    with col2:
        if st.button("Run Optimization →", type="primary"):
            st.session_state.step = 4
            st.rerun()


def show_optimization_step():
    """Step 4: Run optimization and show results."""
    st.header("Step 4: Optimization Results")
    
    classes_df = st.session_state.classes_df
    tutors_df = st.session_state.tutors_df
    preferences = st.session_state.preferences
    max_classes = st.session_state.max_classes
    
    # Build preference dictionary for LP
    pref_dict = {}
    for tutor, courses in preferences.items():
        for course in courses:
            pref_dict[(tutor, course)] = 10  # All preferences = 10
    
    with st.spinner("🔄 Running Linear Programming optimization..."):
        try:
            # Create and solve LP
            lp = TutorAssignmentLP(
                classes_df=classes_df,
                tutors_df=tutors_df,
                preferences=pref_dict,
                tutor_max_classes=max_classes
            )
            
            solution = lp.solve()
            
            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Status", solution['status'])
            with col2:
                if solution['objective_value']:
                    st.metric("Objective Value", f"{solution['objective_value']:.0f}")
            with col3:
                st.metric("Unassigned Classes", len(solution['unassigned_classes']))
            
            if solution['status'] == 'Optimal':
                st.success("✅ Optimization completed successfully!")
                
                # Create results dataframe
                results_data = []
                for idx, row in classes_df.iterrows():
                    course = row['course']
                    class_id = row['class_id']
                    assigned_tutor = solution['assignments'].get((course, class_id), 'UNASSIGNED')
                    
                    results_data.append({
                        'Course': course,
                        'Class ID': class_id,
                        'Type': row['type'],
                        'Section': row['section'],
                        'Time': row['time'],
                        'Assigned Tutor': assigned_tutor
                    })
                
                results_df = pd.DataFrame(results_data)
                
                # Show assignment table
                st.subheader("📋 Class Assignments")
                
                # Filter options
                col1, col2 = st.columns(2)
                with col1:
                    filter_course = st.selectbox(
                        "Filter by Course:",
                        options=['All'] + sorted(classes_df['course'].unique().tolist())
                    )
                with col2:
                    filter_tutor = st.selectbox(
                        "Filter by Tutor:",
                        options=['All'] + sorted([t for t in tutors_df['tutor_name'].unique()])
                    )
                
                display_df = results_df.copy()
                if filter_course != 'All':
                    display_df = display_df[display_df['Course'] == filter_course]
                if filter_tutor != 'All':
                    display_df = display_df[display_df['Assigned Tutor'] == filter_tutor]
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Tutor workload summary
                st.subheader("👥 Tutor Workload Summary")
                
                workload_data = []
                for tutor in tutors_df['tutor_name'].unique():
                    load = solution['tutor_loads'][tutor]
                    assigned_classes = load['classes']
                    
                    # Group by course
                    courses_count = {}
                    for course, class_id in assigned_classes:
                        courses_count[course] = courses_count.get(course, 0) + 1
                    
                    courses_str = ', '.join([f"{course}({count})" for course, count in courses_count.items()])
                    
                    workload_data.append({
                        'Tutor': tutor,
                        'Total Classes': load['total'],
                        'Max Allowed': max_classes.get(tutor, 0),
                        'Utilization': f"{load['total']}/{max_classes.get(tutor, 0)}",
                        'Courses Assigned': courses_str if courses_str else 'None'
                    })
                
                workload_df = pd.DataFrame(workload_data)
                workload_df = workload_df.sort_values('Total Classes', ascending=False)
                st.dataframe(workload_df, use_container_width=True, hide_index=True)
                
                # Visualization
                fig_workload = px.bar(
                    workload_df,
                    x='Tutor',
                    y='Total Classes',
                    title='Tutor Workload Distribution',
                    color='Total Classes',
                    color_continuous_scale='Blues'
                )
                fig_workload.update_layout(height=400, showlegend=False, xaxis_tickangle=45)
                st.plotly_chart(fig_workload, use_container_width=True)
                
                # Unassigned classes
                if solution['unassigned_classes']:
                    st.subheader("⚠️ Unassigned Classes")
                    st.error(f"The following {len(solution['unassigned_classes'])} classes could not be assigned:")
                    
                    unassigned_data = []
                    for course, class_id in solution['unassigned_classes']:
                        class_row = classes_df[
                            (classes_df['course'] == course) & 
                            (classes_df['class_id'] == class_id)
                        ].iloc[0]
                        
                        # Find why it couldn't be assigned
                        qualified_tutors = [t for t, courses in preferences.items() if course in courses]
                        
                        unassigned_data.append({
                            'Course': course,
                            'Class ID': class_id,
                            'Section': class_row['section'],
                            'Time': class_row['time'],
                            'Qualified Tutors': len(qualified_tutors),
                            'Possible Reason': 'Time conflict or capacity exceeded'
                        })
                    
                    unassigned_df = pd.DataFrame(unassigned_data)
                    st.dataframe(unassigned_df, use_container_width=True, hide_index=True)
                else:
                    st.success("🎉 All classes successfully assigned!")
                
                # Download results
                st.subheader("📥 Download Results")
                
                # Create Excel output
                from io import BytesIO
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    results_df.to_excel(writer, sheet_name='Assignments', index=False)
                    workload_df.to_excel(writer, sheet_name='Tutor Workload', index=False)
                    if solution['unassigned_classes']:
                        unassigned_df.to_excel(writer, sheet_name='Unassigned', index=False)
                
                st.download_button(
                    label="📥 Download Results (Excel)",
                    data=output.getvalue(),
                    file_name="tutor_assignments_T3.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
            else:
                st.error(f"❌ Optimization failed: {solution['status']}")
                st.write("Possible reasons:")
                st.write("- Insufficient tutor capacity")
                st.write("- Too many time conflicts")
                st.write("- No qualified tutors for some courses")
            
        except Exception as e:
            st.error(f"❌ Error during optimization: {str(e)}")
            st.exception(e)
    
    # Navigation
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back to Analysis"):
            st.session_state.step = 3
            st.rerun()
    
    with col2:
        if st.button("Start Over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


if __name__ == "__main__":
    main()
