from optparse import OptionParser
from predict import Predict

parser = OptionParser()
parser.add_option("-n", "--name",       dest="name",        help="Name", metavar="NAME")
parser.add_option("-m", "--model_id",   dest="model_id",    help="Model ID", metavar="MODELID")
parser.add_option("-x", "--secret",     dest="secret",      help="Secret", metavar="SECRET")
parser.add_option("-f", "--formatter",  dest="formatter",   help="Formatter", metavar="FORMATTER")
parser.add_option("-s", "--symbol",     dest="symbol",      help="Symbol", metavar="NAME")
parser.add_option("-b", "--base_url",   dest="base_url",    help="Base url", metavar="NAME")



(options, args) = parser.parse_args()

print(options)


predict = Predict(options.name, options.model_id, options.secret,  options.formatter, options.symbol, options.base_url)
predict.run()