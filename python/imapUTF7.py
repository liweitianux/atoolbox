#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# This code was originally in PloneMailList, a GPL'd software.
# http://svn.plone.org/svn/collective/mxmImapClient/trunk/imapUTF7.py
# http://bugs.python.org/issue5305
#
# Port to Python 3.x
# Credit: https://github.com/MarechJ/py3_imap_utf7
#
# 2016-01-23
# Aaron LI
#

"""
Imap folder names are encoded using a special version of utf-7 as defined in RFC
2060 section 5.1.3.

5.1.3.  Mailbox International Naming Convention

   By convention, international mailbox names are specified using a
   modified version of the UTF-7 encoding described in [UTF-7].  The
   purpose of these modifications is to correct the following problems
   with UTF-7:

      1) UTF-7 uses the "+" character for shifting; this conflicts with
         the common use of "+" in mailbox names, in particular USENET
         newsgroup names.

      2) UTF-7's encoding is BASE64 which uses the "/" character; this
         conflicts with the use of "/" as a popular hierarchy delimiter.

      3) UTF-7 prohibits the unencoded usage of "\"; this conflicts with
         the use of "\" as a popular hierarchy delimiter.

      4) UTF-7 prohibits the unencoded usage of "~"; this conflicts with
         the use of "~" in some servers as a home directory indicator.

      5) UTF-7 permits multiple alternate forms to represent the same
         string; in particular, printable US-ASCII chararacters can be
         represented in encoded form.

   In modified UTF-7, printable US-ASCII characters except for "&"
   represent themselves; that is, characters with octet values 0x20-0x25
   and 0x27-0x7e.  The character "&" (0x26) is represented by the two-
   octet sequence "&-".

   All other characters (octet values 0x00-0x1f, 0x7f-0xff, and all
   Unicode 16-bit octets) are represented in modified BASE64, with a
   further modification from [UTF-7] that "," is used instead of "/".
   Modified BASE64 MUST NOT be used to represent any printing US-ASCII
   character which can represent itself.

   "&" is used to shift to modified BASE64 and "-" to shift back to US-
   ASCII.  All names start in US-ASCII, and MUST end in US-ASCII (that
   is, a name that ends with a Unicode 16-bit octet MUST end with a "-
   ").

      For example, here is a mailbox name which mixes English, Japanese,
      and Chinese text: ~peter/mail/&ZeVnLIqe-/&U,BTFw-
"""


import binascii
import codecs


## encoding

def modified_base64(s:str):
    s = s.encode('utf-16be')  # UTF-16, big-endian byte order
    return binascii.b2a_base64(s).rstrip(b'\n=').replace(b'/', b',')

def doB64(_in, r):
    if _in:
        r.append(b'&' + modified_base64(''.join(_in)) + b'-')
        del _in[:]

def encoder(s:str):
    r = []
    _in = []
    for c in s:
        ordC = ord(c)
        if 0x20 <= ordC <= 0x25 or 0x27 <= ordC <= 0x7e:
            doB64(_in, r)
            r.append(c.encode())
        elif c == '&':
            doB64(_in, r)
            r.append(b'&-')
        else:
            _in.append(c)
    doB64(_in, r)
    return (b''.join(r), len(s))


## decoding

def modified_unbase64(s:bytes):
    b = binascii.a2b_base64(s.replace(b',', b'/') + b'===')
    return b.decode('utf-16be')

def decoder(s:bytes):
    r = []
    decode = bytearray()
    for c in s:
        if c == ord('&') and not decode:
            decode.append(ord('&'))
        elif c == ord('-') and decode:
            if len(decode) == 1:
                r.append('&')
            else:
                r.append(modified_unbase64(decode[1:]))
            decode = bytearray()
        elif decode:
            decode.append(c)
        else:
            r.append(chr(c))
    if decode:
        r.append(modified_unbase64(decode[1:]))
    bin_str = ''.join(r)
    return (bin_str, len(s))


class StreamReader(codecs.StreamReader):
    def decode(self, s, errors='strict'):
        return decoder(s)


class StreamWriter(codecs.StreamWriter):
    def decode(self, s, errors='strict'):
        return encoder(s)


def imap4_utf_7(name):
    if name == 'imap4-utf-7':
        return (encoder, decoder, StreamReader, StreamWriter)


codecs.register(imap4_utf_7)


## testing methods

def imapUTF7Encode(ust):
    "Returns imap utf-7 encoded version of string"
    return ust.encode('imap4-utf-7')

def imapUTF7EncodeSequence(seq):
    "Returns imap utf-7 encoded version of strings in sequence"
    return [imapUTF7Encode(itm) for itm in seq]


def imapUTF7Decode(st):
    "Returns utf7 encoded version of imap utf-7 string"
    return st.decode('imap4-utf-7')

def imapUTF7DecodeSequence(seq):
    "Returns utf7 encoded version of imap utf-7 strings in sequence"
    return [imapUTF7Decode(itm) for itm in seq]


def utf8Decode(st):
    "Returns utf7 encoded version of imap utf-7 string"
    return st.decode('utf-8')


def utf7SequenceToUTF8(seq):
    "Returns utf7 encoded version of imap utf-7 strings in sequence"
    return [itm.decode('imap4-utf-7').encode('utf-8') for itm in seq]


__all__ = [ 'imapUTF7Encode', 'imapUTF7Decode' ]


if __name__ == '__main__':
    testdata = [
            (u'foo\r\n\nbar\n', b'foo&AA0ACgAK-bar&AAo-'),
            (u'测试', b'&bUuL1Q-'),
            (u'Hello 世界', b'Hello &ThZ1TA-')
    ]
    for s, e in testdata:
        #assert s == decoder(encoder(s)[0])[0]
        assert s == imapUTF7Decode(e)
        assert e == imapUTF7Encode(s)
        assert s == imapUTF7Decode(imapUTF7Encode(s))
        assert e == imapUTF7Encode(imapUTF7Decode(e))
    print("All tests passed!")

#  vim: set ts=4 sw=4 tw=0 fenc=utf-8 ft=python: #
