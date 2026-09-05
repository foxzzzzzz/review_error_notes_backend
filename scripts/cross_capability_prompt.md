判断所给局部图片中 target_bbox_normalized 指定框是否包含真实红色叉号的交点。
只看指定框里的目标，不要因上下文里另有真叉就判断为 cross。
cross：红色批改叉号；not_cross：勾、圈、印刷红线、汉字交叉笔画、页码等；无法确定用 uncertain。
不需要解题、定位题框或读取学生答案。evidence 简短描述可观察的形状、颜色证据。
只返回符合 Schema 的 JSON，question_id 原样返回，恰好一项。
