"""empty message

Revision ID: 63dba2060f71
Revises: None
Create Date: 2016-03-14 21:34:11.940277

"""

# revision identifiers, used by Alembic.
revision = '63dba2060f71'
down_revision = None

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

def upgrade():
    ### commands auto generated by Alembic - please adjust! ###
    op.create_table('NudeDetect',
    sa.Column('image_id', sa.Integer(), nullable=False),
    sa.Column('image_url', sa.String(), nullable=True),
    sa.Column('image_timestamp', sa.date(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    ### end Alembic commands ###


def downgrade():
    ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('NudeDetect')
    ### end Alembic commands ###
