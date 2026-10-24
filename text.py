def getFontSize(font, text):
    sprite = font.render(text, True, (0, 0, 0))  # Rendered once; color doesn't affect size
    return [sprite.get_width(), sprite.get_height()]


def simpleText(screen, color, font, x, y, text):
    screen.blit(font.render(text, True, color), (x, y))


def wrapText(font, text, maxLen):
    words, line, wrappedLines = text.split(), "", []
    for word in words:
        testLine = f"{line} {word}".strip()
        if font.size(testLine)[0] > maxLen:
            wrappedLines.append(line)
            line = word
        else:
            line = testLine
    wrappedLines.append(line)
    maxWidth = max(font.size(line)[0] for line in wrappedLines)
    return wrappedLines, maxWidth


def drawText(screen, color, font, x, y, text, color2=None, shadowSize=0, wrap=False, maxLen=None, antiAliasing=False, justify="left", centeredVertically=False):
    def drawLine(currentLine, xOffset, yOffset):
        shadowPos = (xOffset + shadowSize, yOffset + shadowSize)
        textPos = (xOffset, yOffset)
        if shadowSize and color2:
            screen.blit(font.render(currentLine, antiAliasing, color2), shadowPos)
        screen.blit(font.render(currentLine, antiAliasing, color), textPos)

    if wrap and maxLen:
        lines, _ = wrapText(font, text, maxLen)
    else:
        lines = [text]

    totalHeight = len(lines) * font.get_height() * 1.1
    baseY = y - (totalHeight / 2 if centeredVertically else 0)

    for i, line in enumerate(lines):
        lineWidth = font.size(line)[0]
        if justify == "middle" or justify == "center":
            baseX = x - (lineWidth / 2)
        elif justify == "right":
            baseX = x - lineWidth
        else:  # Default to left
            baseX = x
        drawLine(line, baseX, baseY + i * font.get_height() * 1.1)
