import sys
from discord import Webhook, RequestsWebhookAdapter

webhook = Webhook.from_url("https://discord.com/api/webhooks/1000055986528198767/sZhup-kBr9wqVxIN4vDb5sRUJ9D-7mXaSeZxWssmprWiMqeC3KbmeNGiDoIuyZU4lgWA", adapter=RequestsWebhookAdapter())
webhook.send(' '.join(sys.argv[1:]))