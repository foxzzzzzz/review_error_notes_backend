你正在查看一张经过标准化处理的完整作业页面。页面中老师用红圈、红叉或其他明确批改标记指出了错题。

找出每一道被批改的独立错题。只返回严格 JSON，不要解释，不要 Markdown：
{
  "wrong_questions": [
    {
      "printed_question": "图片中直接可见的印刷题面或学习对象",
      "student_answer": "图片中直接可见的学生作答",
      "model_bbox": [0.10, 0.20, 0.30, 0.40],
      "teacher_mark_type": "red_cross",
      "teacher_mark_bbox": [0.24, 0.18, 0.32, 0.28],
      "confidence": 0.95,
      "uncertain_fields": []
    }
  ]
}

要求：
1. 只返回有明确红色教师批改标记的错题。黑色或灰色的铅笔勾选、学生书写笔画、印刷符号不能单独作为错题依据；没有错题时返回 {"wrong_questions":[]}。
2. printed_question 和 student_answer 只抄录图片中直接可见的内容；看不清时使用 null，不得猜测、补全或纠正。
3. 每个被批改的答题格或选择项分别返回一个条目，每道独立错题只返回一次。同一行存在多个独立红色批改标记时，必须分别返回多个条目；禁止把整行多个词语合并为一个条目。
4. model_bbox 必须覆盖单个完整最小独立作答单元，不能覆盖同一行的其他答题格。坐标采用相对于整张页面的归一化 [left, top, right, bottom]，范围为 0 到 1。
5. teacher_mark_type 只能是 red_circle、red_cross 或 red_other；teacher_mark_bbox 必须只覆盖支持当前错题的红色教师批改标记，不能指向黑色或灰色痕迹。teacher_mark_bbox 使用与 model_bbox 相同的归一化坐标格式。
6. confidence 为 0 到 1 的数值。uncertain_fields 只能列出 printed_question、student_answer 或 model_bbox 中看不清或不可靠的字段。
