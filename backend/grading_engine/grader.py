import sympy as sp
from sympy import symbols, Eq, solve, simplify, latex
import re
import numpy as np
from difflib import SequenceMatcher
import json

class AutomaticGrader:
    """Automated grading system for math problems"""
    
    def __init__(self):
        self.load_answer_key()
    
    def load_answer_key(self):
        """Load correct answers for the probability paper"""
        # Based on the provided probability paper
        self.answer_key = {
            'Q1A': {
                'type': 'regression',
                'expected_answers': {
                    'regression_y_on_x': 'y = 0.613x + 13.87',
                    'regression_x_on_y': 'x = 1.178y - 15.43',
                    'correlation_coefficient': '0.863'
                },
                'points': 5,
                'partial_credit': True
            },
            'Q1B': {
                'type': 'regression_prediction',
                'expected_answers': {
                    'sales_estimate': '40.6',
                    'advertisement_expenditure': '8.33'
                },
                'points': 5,
                'partial_credit': True
            },
            'Q2A': {
                'type': 'hypothesis_test',
                'expected_answers': {
                    'z_statistic': '2.236',
                    'conclusion': 'reject null hypothesis'
                },
                'points': 5,
                'partial_credit': True
            },
            'Q2B': {
                'type': 'proportion_test',
                'expected_answers': {
                    'z_statistic': '1.897',
                    'p_value': '0.0289',
                    'conclusion': 'significant improvement'
                },
                'points': 5,
                'partial_credit': True
            }
        }
    
    def similarity_score(self, text1, text2):
        """Calculate similarity between two text strings"""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def extract_numbers(self, text):
        """Extract all numbers from text"""
        pattern = r'-?\d+\.?\d*'
        numbers = re.findall(pattern, text)
        return [float(num) for num in numbers]
    
    def is_equation_equivalent(self, student_eq, expected_eq, tolerance=0.01):
        """Check if two equations are mathematically equivalent"""
        try:
            # Parse equations using sympy
            x, y = symbols('x y')
            
            # Clean and parse equations
            student_eq = student_eq.replace('=', '-(').replace(' ', '') + ')'
            expected_eq = expected_eq.replace('=', '-(').replace(' ', '') + ')'
            
            student_expr = sp.sympify(student_eq)
            expected_expr = sp.sympify(expected_eq)
            
            # Check if difference is close to zero
            diff = simplify(student_expr - expected_expr)
            
            # Test with several x values
            test_values = [0, 1, 5, 10, -1, -5]
            for val in test_values:
                try:
                    diff_val = float(diff.subs(x, val))
                    if abs(diff_val) > tolerance:
                        return False
                except:
                    continue
            
            return True
            
        except Exception as e:
            print(f"Equation comparison error: {e}")
            return False
    
    def is_number_close(self, student_num, expected_num, tolerance=0.05):
        """Check if numbers are close within tolerance"""
        try:
            student_val = float(student_num)
            expected_val = float(expected_num)
            
            relative_error = abs(student_val - expected_val) / abs(expected_val)
            return relative_error <= tolerance
            
        except:
            return False
    
    def grade_regression_problem(self, student_answer, question_key):
        """Grade regression analysis problems"""
        expected = question_key['expected_answers']
        max_points = question_key['points']
        
        score = 0
        feedback = []
        details = {}
        
        # Extract student's numerical answers
        student_numbers = self.extract_numbers(student_answer)
        
        # Check regression equation coefficients
        if 'regression_y_on_x' in expected:
            expected_eq = expected['regression_y_on_x']
            
            # Look for equation pattern in student answer
            eq_pattern = r'y\s*=\s*([-+]?\d*\.?\d*)\s*x\s*([-+]\s*\d*\.?\d*)'
            match = re.search(eq_pattern, student_answer.lower())
            
            if match:
                try:
                    slope = float(match.group(1))
                    intercept = float(match.group(2).replace(' ', ''))
                    student_eq = f"y = {slope}x + {intercept}"
                    
                    if self.is_equation_equivalent(student_eq, expected_eq):
                        score += max_points * 0.4  # 40% for correct equation
                        feedback.append("✓ Regression equation is correct")
                        details['regression_y_on_x'] = {'correct': True, 'student': student_eq}
                    else:
                        score += max_points * 0.2  # Partial credit for attempt
                        feedback.append("⚠ Regression equation has errors")
                        details['regression_y_on_x'] = {'correct': False, 'student': student_eq}
                except:
                    feedback.append("✗ Could not parse regression equation")
                    details['regression_y_on_x'] = {'correct': False, 'student': student_answer[:50]}
            else:
                feedback.append("✗ Regression equation not found")
        
        # Check correlation coefficient
        if 'correlation_coefficient' in expected:
            expected_r = float(expected['correlation_coefficient'])
            
            # Look for correlation coefficient
            for num in student_numbers:
                if self.is_number_close(num, expected_r):
                    score += max_points * 0.3  # 30% for correlation
                    feedback.append("✓ Correlation coefficient is correct")
                    details['correlation_coefficient'] = {'correct': True, 'student': num}
                    break
            else:
                feedback.append("⚠ Correlation coefficient incorrect or not found")
                details['correlation_coefficient'] = {'correct': False}
        
        return {
            'score': min(score, max_points),
            'max_score': max_points,
            'percentage': min(score / max_points * 100, 100),
            'feedback': feedback,
            'details': details
        }
    
    def grade_hypothesis_test(self, student_answer, question_key):
        """Grade hypothesis testing problems"""
        expected = question_key['expected_answers']
        max_points = question_key['points']
        
        score = 0
        feedback = []
        details = {}
        
        student_numbers = self.extract_numbers(student_answer)
        student_text = student_answer.lower()
        
        # Check z-statistic
        if 'z_statistic' in expected:
            expected_z = float(expected['z_statistic'])
            
            for num in student_numbers:
                if self.is_number_close(num, expected_z):
                    score += max_points * 0.5  # 50% for test statistic
                    feedback.append("✓ Test statistic is correct")
                    details['z_statistic'] = {'correct': True, 'student': num}
                    break
            else:
                feedback.append("⚠ Test statistic incorrect or not found")
                details['z_statistic'] = {'correct': False}
        
        # Check conclusion
        if 'conclusion' in expected:
            expected_conclusion = expected['conclusion'].lower()
            
            conclusion_keywords = {
                'reject': ['reject', 'rejected'],
                'fail to reject': ['fail to reject', 'accept', 'do not reject'],
                'significant': ['significant', 'significance'],
                'not significant': ['not significant', 'insignificant']
            }
            
            conclusion_found = False
            if 'reject' in expected_conclusion:
                if any(keyword in student_text for keyword in conclusion_keywords['reject']):
                    score += max_points * 0.3  # 30% for correct conclusion
                    feedback.append("✓ Conclusion is correct")
                    details['conclusion'] = {'correct': True}
                    conclusion_found = True
            
            if not conclusion_found:
                feedback.append("⚠ Conclusion incorrect or unclear")
                details['conclusion'] = {'correct': False}
        
        return {
            'score': min(score, max_points),
            'max_score': max_points,
            'percentage': min(score / max_points * 100, 100),
            'feedback': feedback,
            'details': details
        }
    
    def grade_answer(self, question_id, student_answer):
        """Grade a specific answer"""
        if question_id not in self.answer_key:
            return {
                'score': 0,
                'max_score': 5,
                'percentage': 0,
                'feedback': ["Question not found in answer key"],
                'details': {}
            }
        
        question_key = self.answer_key[question_id]
        question_type = question_key['type']
        
        # Route to appropriate grading method
        if question_type in ['regression', 'regression_prediction']:
            return self.grade_regression_problem(student_answer, question_key)
        elif question_type in ['hypothesis_test', 'proportion_test']:
            return self.grade_hypothesis_test(student_answer, question_key)
        else:
            # Generic grading
            return self.grade_generic(student_answer, question_key)
    
    def grade_generic(self, student_answer, question_key):
        """Generic grading for other question types"""
        max_points = question_key['points']
        expected = question_key['expected_answers']
        
        score = 0
        feedback = []
        
        # Simple keyword matching
        student_text = student_answer.lower()
        
        for key, expected_val in expected.items():
            if isinstance(expected_val, str):
                similarity = self.similarity_score(student_text, expected_val)
                if similarity > 0.7:
                    score += max_points * 0.8 / len(expected)
                    feedback.append(f"✓ {key} looks correct")
                elif similarity > 0.4:
                    score += max_points * 0.4 / len(expected)
                    feedback.append(f"⚠ {key} partially correct")
        
        return {
            'score': min(score, max_points),
            'max_score': max_points,
            'percentage': min(score / max_points * 100, 100),
            'feedback': feedback,
            'details': {}
        }
    
    def grade_complete_paper(self, ocr_results):
        """Grade complete paper from OCR results"""
        total_score = 0
        max_total_score = 0
        question_grades = {}
        
        # Map OCR regions to questions
        question_mapping = self.map_regions_to_questions(ocr_results)
        
        for question_id, student_answer in question_mapping.items():
            grade_result = self.grade_answer(question_id, student_answer)
            question_grades[question_id] = grade_result
            
            total_score += grade_result['score']
            max_total_score += grade_result['max_score']
        
        return {
            'total_score': total_score,
            'max_total_score': max_total_score,
            'percentage': (total_score / max_total_score * 100) if max_total_score > 0 else 0,
            'question_grades': question_grades,
            'summary': self.generate_summary(question_grades)
        }
    
    def map_regions_to_questions(self, ocr_results):
        """Map OCR regions to specific questions"""
        question_mapping = {}
        
        # Simple heuristic: assume regions are in order
        question_counter = 1
        part_counter = 'A'
        
        for i, region in enumerate(ocr_results.get('regions', [])):
            if region['type'] == 'answer':
                question_id = f"Q{question_counter}{part_counter}"
                question_mapping[question_id] = region.get('text', '')
                
                # Advance counters
                if part_counter == 'A':
                    part_counter = 'B'
                else:
                    part_counter = 'A'
                    question_counter += 1
        
        return question_mapping
    
    def generate_summary(self, question_grades):
        """Generate grading summary"""
        total_questions = len(question_grades)
        correct_questions = sum(1 for grade in question_grades.values() 
                              if grade['percentage'] >= 70)
        
        summary = {
            'total_questions': total_questions,
            'correct_questions': correct_questions,
            'accuracy': (correct_questions / total_questions * 100) if total_questions > 0 else 0,
            'strengths': [],
            'improvements': []
        }
        
        # Analyze performance
        for q_id, grade in question_grades.items():
            if grade['percentage'] >= 80:
                summary['strengths'].append(f"Strong performance in {q_id}")
            elif grade['percentage'] < 50:
                summary['improvements'].append(f"Need improvement in {q_id}")
        
        return summary
