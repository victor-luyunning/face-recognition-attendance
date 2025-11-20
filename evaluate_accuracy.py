"""
人脸识别考勤系统准确率评估工具

使用方法:
1. 在test_data文件夹中准备测试照片
2. 创建ground_truth.json文件,标注每张照片中的学生ID
3. 运行此脚本进行评估

Ground Truth格式示例:
{
    "photo1.jpg": [1, 2, 3],  # 照片中包含学生ID 1, 2, 3
    "photo2.jpg": [2, 4, 5],
    ...
}
"""

import json
import sqlite3
import cv2
import numpy as np
import face_recognition
from pathlib import Path
from datetime import datetime

class AccuracyEvaluator:
    def __init__(self, db_path="students.db", class_id=1):
        self.db_path = db_path
        self.class_id = class_id
        self.threshold = 0.42  # 识别阈值
        
    def load_ground_truth(self, json_path):
        """加载真实标签"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_face_features(self, image):
        """提取人脸特征(与主系统保持一致)"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_image, model='hog')
            
            if not face_locations:
                face_locations = face_recognition.face_locations(rgb_image, model='cnn')
            
            if not face_locations:
                enhanced = cv2.convertScaleAbs(rgb_image, alpha=1.2, beta=30)
                face_locations = face_recognition.face_locations(enhanced, model='hog', number_of_times_to_upsample=2)
                if face_locations:
                    rgb_image = enhanced
            
            if not face_locations:
                return [], []
            
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            return face_encodings, face_locations
        except Exception as e:
            print(f"提取人脸特征时出错: {str(e)}")
            return [], []
    
    def recognize_photo(self, photo_path, students):
        """识别一张照片中的学生"""
        # 读取照片
        image = cv2.imread(photo_path)
        if image is None:
            print(f"无法读取照片: {photo_path}")
            return []
        
        # 提取人脸特征
        face_encodings, _ = self.extract_face_features(image)
        if not face_encodings:
            print(f"照片中未检测到人脸: {photo_path}")
            return []
        
        # 识别每张人脸
        recognized_ids = []
        for face_encoding in face_encodings:
            best_match_id = None
            best_similarity = 0
            
            for student_id, name, stored_encoding in students:
                try:
                    stored_encoding_array = np.frombuffer(stored_encoding, dtype=np.float64)
                    face_distance = face_recognition.face_distance([stored_encoding_array], face_encoding)[0]
                    similarity = 1 - face_distance
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match_id = student_id
                except Exception:
                    continue
            
            if best_match_id and best_similarity >= self.threshold:
                if best_match_id not in recognized_ids:
                    recognized_ids.append(best_match_id)
        
        return recognized_ids
    
    def evaluate(self, test_dir, ground_truth_path):
        """执行评估"""
        # 加载真实标签
        ground_truth = self.load_ground_truth(ground_truth_path)
        
        # 加载学生数据
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT id, name, face_encoding FROM students WHERE class_id = ?", (self.class_id,))
        students = c.fetchall()
        conn.close()
        
        if not students:
            print("数据库中没有学生数据")
            return
        
        # 评估指标
        total_true_positives = 0  # 正确识别
        total_false_positives = 0  # 误识别
        total_false_negatives = 0  # 漏识别
        total_actual_students = 0  # 实际学生总数
        
        results = []
        
        # 对每张测试照片进行评估
        for photo_name, true_ids in ground_truth.items():
            photo_path = Path(test_dir) / photo_name
            
            if not photo_path.exists():
                print(f"照片不存在: {photo_path}")
                continue
            
            # 系统识别结果
            predicted_ids = self.recognize_photo(str(photo_path), students)
            
            # 计算指标
            true_ids_set = set(true_ids)
            predicted_ids_set = set(predicted_ids)
            
            tp = len(true_ids_set & predicted_ids_set)  # 正确识别
            fp = len(predicted_ids_set - true_ids_set)  # 误识别
            fn = len(true_ids_set - predicted_ids_set)  # 漏识别
            
            total_true_positives += tp
            total_false_positives += fp
            total_false_negatives += fn
            total_actual_students += len(true_ids)
            
            # 计算单张照片的准确率
            photo_precision = tp / len(predicted_ids) if predicted_ids else 0
            photo_recall = tp / len(true_ids) if true_ids else 0
            photo_f1 = 2 * photo_precision * photo_recall / (photo_precision + photo_recall) if (photo_precision + photo_recall) > 0 else 0
            
            results.append({
                'photo': photo_name,
                'true_ids': true_ids,
                'predicted_ids': predicted_ids,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'precision': photo_precision,
                'recall': photo_recall,
                'f1': photo_f1
            })
            
            print(f"\n照片: {photo_name}")
            print(f"  实际学生: {true_ids}")
            print(f"  识别结果: {predicted_ids}")
            print(f"  正确: {tp}, 误识: {fp}, 漏识: {fn}")
            print(f"  精确率: {photo_precision:.2%}, 召回率: {photo_recall:.2%}, F1: {photo_f1:.2%}")
        
        # 计算总体指标
        overall_precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
        overall_recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        overall_accuracy = total_true_positives / total_actual_students if total_actual_students > 0 else 0
        
        print("\n" + "="*60)
        print("总体评估结果")
        print("="*60)
        print(f"测试照片数: {len(results)}")
        print(f"实际学生总数: {total_actual_students}")
        print(f"正确识别: {total_true_positives}")
        print(f"误识别: {total_false_positives}")
        print(f"漏识别: {total_false_negatives}")
        print(f"\n精确率 (Precision): {overall_precision:.2%}")
        print(f"召回率 (Recall): {overall_recall:.2%}")
        print(f"F1分数: {overall_f1:.2%}")
        print(f"准确率 (Accuracy): {overall_accuracy:.2%}")
        print("="*60)
        
        # 保存评估报告
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'threshold': self.threshold,
            'overall_metrics': {
                'precision': overall_precision,
                'recall': overall_recall,
                'f1_score': overall_f1,
                'accuracy': overall_accuracy,
                'total_photos': len(results),
                'total_students': total_actual_students,
                'true_positives': total_true_positives,
                'false_positives': total_false_positives,
                'false_negatives': total_false_negatives
            },
            'detailed_results': results
        }
        
        report_path = f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n评估报告已保存到: {report_path}")
        
        return report


if __name__ == "__main__":
    print("="*60)
    print("人脸识别考勤系统准确率评估工具")
    print("="*60)
    
    # 配置参数
    TEST_DIR = "test_data"  # 测试照片目录
    GROUND_TRUTH = "ground_truth.json"  # 真实标签文件
    CLASS_ID = 1  # 班级ID
    
    # 检查文件是否存在
    if not Path(TEST_DIR).exists():
        print(f"\n错误: 测试目录不存在: {TEST_DIR}")
        print("请创建test_data文件夹并放入测试照片")
        exit(1)
    
    if not Path(GROUND_TRUTH).exists():
        print(f"\n错误: 真实标签文件不存在: {GROUND_TRUTH}")
        print("\n请创建ground_truth.json文件,格式示例:")
        print(json.dumps({
            "photo1.jpg": [1, 2, 3],
            "photo2.jpg": [2, 4, 5]
        }, ensure_ascii=False, indent=2))
        exit(1)
    
    # 执行评估
    evaluator = AccuracyEvaluator(class_id=CLASS_ID)
    evaluator.evaluate(TEST_DIR, GROUND_TRUTH)
