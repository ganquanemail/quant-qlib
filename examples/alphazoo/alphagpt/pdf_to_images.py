import fitz  # PyMuPDF

def pdf_to_images(pdf_path, output_folder):
    # 打开PDF文件
    pdf_document = fitz.open(pdf_path)
    # 遍历每一页
    for page_number in range(len(pdf_document)):
        # 获取当前页
        page = pdf_document[page_number]
        # 将页面渲染为图像
        pix = page.get_pixmap()
        # 为每页生成一个唯一的文件名
        output = f"{output_folder}/factorcalander2024_page{page_number}.png"
        # 保存图像
        pix.save(output)
    # 关闭PDF文件
    pdf_document.close()

# 使用函数
# pdf_to_images('path_to_your_pdf.pdf', 'output_folder')
if __name__ == '__main__':

    pdf_path = 'data/factorcalander/factorcalander2024.pdf'
    output_folder = 'data/factorcalander'

    pdf_to_images(pdf_path, output_folder)