图片为错题图，其中错题被人为用红笔做了批改：红圈和红叉标记了错题，每一个红圈+红叉对应一个错题。

你基于上面的信息，把错题区域和内容识别出来。

请只返回严格 JSON，不要解释，不要 Markdown。格式如下：
{
  "wrong_questions": [
    {
      "printed_question": "图片中直接可见的印刷题面",
      "student_answer": "图片中直接可见的学生作答",
      "bbox": [0.10, 0.20, 0.30, 0.40],
      "confidence": 0.95
    }
  ]
}

要求：
1. 每个红圈和红叉共同对应的独立错题只返回一次。
2. bbox 覆盖该错题的完整最小独立作答单元，不得只框红圈或红叉。
3. bbox 相对于整张原图，采用归一化 [left, top, right, bottom] 坐标，范围为 0 到 1。
4. printed_question 和 student_answer 只抄录图片中直接可见的内容，看不清时使用 null，不得猜测或自动纠正。
5. 没有识别到错题时返回 {"wrong_questions":[]}。
