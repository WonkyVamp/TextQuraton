import re
info="Hospital\nCare offger\nBILL BOOK\nAatmiya Care Hospital & ICU,\n1,2 Floor, Ujjam Enclave\nAatmiya\nOpp. Chamunda Mata Temple,\nZadeshwar Road, Bharuch.\nM.+91 74052 32292 ff\nService That Heal\nE - mail: aatmiyacarehospital@gmail.com\nName: viralkumur Humur Patel. ................./ ......... 9 Bill No. 832\nAddress: ... krishnet. ..Date ...... 6/03/2022\n..Mo ..................\nDescription Amount Rs.\n1501\nfollowup charge\nChaudhari\nDr. Bhavesh), CIH.\nMD (Medicine\nReg. No. G - 24504\nTOTAL ..... 7501"

regexp_phones= re.compile(r"((\+){0,1}91(\s){0,1}(\-){0,1}(\s){0,1}){0,1}[0-9][0-9](\s){0,1}(\-){0,1}(\s){0,1}[1-9]{1}(\s){0,1}(\-){0,1}(\s){0,1}([0-9]{1}(\s){0,1}(\-){0,1}(\s){0,1}){1,6}[0-9]")
phones=(re.search(regexp_phones, info))
print(phones.group())
