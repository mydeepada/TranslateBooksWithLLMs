# Guide pour les Agents - TranslateBooksWithLLMs

## Docker

### Image officielle

L'image Docker officielle est publiée sur GitHub Container Registry :

```bash
docker pull ghcr.io/hydropix/translatebookswithllms:latest
```

**Note** : Le nom de l'image est `translatebookswithllms` (avec un 's' à la fin), pas `translatebookwithllm`.

### Lancer avec DEFAULT_TARGET_LANGUAGE

Pour reproduire/tester le bug #108 (DEFAULT_TARGET_LANGUAGE ignoré) :

```bash
# Lancer avec Italian comme langue cible par défaut
docker run -d \
  -p 5000:5000 \
  -e DEFAULT_TARGET_LANGUAGE=Italian \
  -e DEBUG_MODE=true \
  -e HOST=0.0.0.0 \
  -e LLM_PROVIDER=ollama \
  -e API_ENDPOINT=http://host.docker.internal:11434/api/generate \
  ghcr.io/hydropix/translatebookswithllms:latest
```

Puis tester dans une fenêtre privée : http://localhost:5000

Le champ "Target Language" doit afficher "Italian" (et non pas la langue du navigateur).

### Docker Compose

```yaml
version: '3.8'

services:
  translate-book:
    image: ghcr.io/hydropix/translatebookswithllms:latest
    ports:
      - "5000:5000"
    environment:
      - DEFAULT_TARGET_LANGUAGE=Italian
      - DEBUG_MODE=true
      - HOST=0.0.0.0
    volumes:
      - ./translated_files:/app/translated_files
      - ./logs:/app/logs
```

### Vérifier les logs

```bash
# Voir les logs du conteneur
docker logs <container_id>

# Vérifier que DEFAULT_TARGET_LANGUAGE est bien chargée
docker logs <container_id> 2>&1 | grep DEFAULT_TARGET_LANGUAGE
```

### Build local

```bash
# Builder l'image localement
docker build -f deployment/Dockerfile -t translatebook:local .

# Lancer l'image locale
docker run -p 5000:5000 -e DEFAULT_TARGET_LANGUAGE=Italian translatebook:local
```

## Workflows GitHub Actions

### Publication Docker

Les images Docker sont buildées et publiées uniquement lors des **tags de version** (`v*.*.*`) :

```bash
# Créer un tag de version
git tag -a v1.0.12 -m "Release v1.0.12"
git push origin v1.0.12
```

Les images suivantes seront créées :
- `ghcr.io/hydropix/translatebookswithllms:latest`
- `ghcr.io/hydropix/translatebookswithllms:1.0.12`
- `ghcr.io/hydropix/translatebookswithllms:1.0`
- `ghcr.io/hydropix/translatebookswithllms:1`

### Workflows disponibles

| Workflow | Fichier | Déclencheur | Description |
|----------|---------|-------------|-------------|
| Docker Publish | `.github/workflows/docker-publish.yml` | Tags `v*.*.*` | Build et push l'image sur ghcr.io |
| Docker Test | `.github/workflows/docker-test.yml` | Push/PR sur `main`/`dev` | Teste que l'image build et démarre (pas de push) |
| Build Windows | `.github/workflows/build-windows.yml` | Tags `v*` | Build l'exécutable Windows |
| Build macOS | `.github/workflows/build-macos.yml` | Tags `v*` | Build l'app macOS |

## Variables d'environnement importantes

| Variable | Description | Défaut |
|----------|-------------|--------|
| `DEFAULT_TARGET_LANGUAGE` | Langue cible par défaut | (vide = auto-détection navigateur) |
| `DEFAULT_SOURCE_LANGUAGE` | Langue source par défaut | (vide = auto-détection) |
| `HOST` | Interface d'écoute | `127.0.0.1` |
| `PORT` | Port du serveur | `5000` |
| `DEBUG_MODE` | Mode debug verbose | `false` |

## Test du bug #108

Pour vérifier que le bug #108 est corrigé :

1. Lancer le conteneur avec `DEFAULT_TARGET_LANGUAGE=Italian`
2. Ouvrir http://localhost:5000 en **fenêtre privée** (pour éviter le cache localStorage)
3. Vérifier que "Target Language" affiche **"Italian"** et non pas la langue du navigateur

Si la langue affichée est celle du navigateur, le bug est présent.
Si c'est "Italian", le bug est corrigé.
