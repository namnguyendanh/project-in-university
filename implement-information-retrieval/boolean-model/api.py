import query


text = """Bắc Triều Tiên doạ cắt đàm phán với Nhật 
Bình Nhưỡng hôm qua ám chỉ khả năng ngừng đàm phán với Tokyo về số phận các con tin Nhật bị bắt cóc vài thập kỷ trước, vì họ cảm thấy việc duy trì liên lạc không còn quan trọng nữa.
Một phát ngôn viên Bộ Ngoại giao CHDCND Triều Tiên cho biết Bình Nhưỡng không thể công nhận hay chấp nhận bản báo cáo mới đây của Tokyo về cuộc điều tra do Bắc Triều Tiên tiến hành, liên quan đến số phận những công dân Nhật từng bị Bình Nhưỡng bắt cóc. 
Tuần trước, Nhật yêu cầu nước này điều tra lại số phận các công dân của họ và tuyên bố sẽ có biện pháp cứng rắn nếu Bình Nhưỡng không phản ứng chân thành và "mau chóng". Tuy nhiên, họ không nói rõ có trừng phạt kinh tế hay không. 
Bắc Triều Tiên cảnh báo sẽ coi các biện pháp trừng phạt kinh tế là một lời tuyên chiến và doạ loại Nhật khỏi vòng đàm phán 6 bên về chương trình hạt nhân của Bình Nhưỡng. 
Hồi tháng 11, CHDCND Tiều Tiên trao trả hài cốt mà theo lời họ là của Megumi Yokota và Kaoru Matsuki, 2 trong số 13 công dân Nhật bị Bình Nhưỡng bắt cóc trong thập kỷ 1970 và 1980. Các xét nghiệm AND cho thấy đây là xương của những người khác. 
"""

result = query.get_result_with_nums(query=text, nums=2)
print(result)
