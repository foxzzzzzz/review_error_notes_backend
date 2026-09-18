你正在查看一张经过标准化处理的完整作业页面。页面中老师用红圈、红叉或其他明确批改标记指出了错题。

找出每一道被批改的独立错题。只返回严格 JSON，不要解释，不要 Markdown：
{
  "wrong_questions": [
    {
      "printed_question": "图片中直接可见的印刷题面或学习对象",
      "student_answer": "图片中直接可见的学生作答",
      "model_bbox": [0.10, 0.20, 0.30, 0.40],
      "confidence": 0.95,
      "uncertain_fields": []
    }
  ]
}

要求：
1. 每道被明确批改的独立错题只返回一次；没有错题时返回 {"wrong_questions":[]}。
2. printed_question 和 student_answer 只抄录图片中直接可见的内容；看不清时使用 null，不得猜测、补全或纠正。
3. model_bbox 必须覆盖完整最小独立作答单元，采用相对于整张页面的归一化 [left, top, right, bottom] 坐标，范围为 0 到 1。
4. confidence 为 0 到 1 的数值。uncertain_fields 只能列出 printed_question、student_answer 或 model_bbox 中看不清或不可靠的字段。
