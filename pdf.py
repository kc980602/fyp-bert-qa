from pdfminer.high_level import extract_text


def extract_text_txt(txt):
    text = txt.replace('\r', '')
    text = text.split('\n\n')
    result = [line for line in text if line.strip() != '']
    return result


def extract_text_pdf(pdf):
    text = extract_text(pdf)
    text = text.replace('-\n', '')
    text = text.replace('\n\n', '[SEP]')
    text = text.replace('\n', ' ')

    text = text.split('[SEP]')

    # result = [line for line in text if len(line.split(' ')) > 8 and line.strip() != '']
    result = [line for line in text if line.strip() != '']

    return result
