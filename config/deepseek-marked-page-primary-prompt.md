你正在查看一张经过标准化处理的完整作业页面。只有老师的红色勾和红色圈可作为指出错题的批改依据。

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
1. 只有红色勾与红色圈是本次认可的老师批改依据。黑色、灰色或其他非红色笔迹（学生书写、铅笔勾选、印刷符号）都不是老师批改，不能据此返回候选；没有错题时返回 {"wrong_questions":[]}。
2. printed_question 和 student_answer 只抄录图片中直接可见的内容；看不清时使用 null，不得猜测、补全或纠正。
3. 每个被老师批改的答题格、填空位或选择项作为一个最小独立候选；同一行多个批改项必须分别返回，禁止整行合并。每道独立错题只返回一次。
4. model_bbox 必须覆盖完整最小独立作答单元，不能覆盖同一行的其他答题格；采用相对于整张页面的归一化 [left, top, right, bottom] 坐标，范围为 0 到 1。
5. confidence 为 0 到 1 的数值。uncertain_fields 只能列出 printed_question、student_answer 或 model_bbox 中看不清或不可靠的字段。
